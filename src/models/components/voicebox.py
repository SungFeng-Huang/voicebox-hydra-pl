import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union
from naturalspeech2_pytorch.aligner import AlignerNet
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from voicebox_pytorch.voicebox_pytorch import AudioEncoderDecoder, VoiceBox as _VB
from voicebox_pytorch.voicebox_pytorch import ConditionalFlowMatcherWrapper as _CFMWrapper
from voicebox_pytorch.voicebox_pytorch import DurationPredictor as _DP
from voicebox_pytorch.voicebox_pytorch import exists, coin_flip, mask_from_frac_lengths, prob_mask_like, rearrange, reduce, curtail_or_pad


class DurationPredictor(_DP):
    """
        1. Override `self.__init__()` to correct aligner, for fixing `self.forward_aligner()`.
            Affecting:
                - `self.forward()`
                    - `self.forward_with_cond_scale()`
                        - `ConditionalFlowMatcherWrapper.sample()`
                        - `ConditionalFlowMatcherWrapper.forward()`
        2. Fix `self.forward()#L823`
            - L823-826: only keep positive condition, since `self_attn_mask` is ensured in L781-782
            - L828: `mask` seems to be corrected into `loss_mask`, while `loss_mask` is declared in L824, so simply remove the if statement
            - L841: `should_align` originally refers to the L818 assertion, therefore should be removed
    """
    def __init__(
        self,
        *,
        audio_enc_dec: AudioEncoderDecoder | None = None,
        tokenizer: Tokenizer | None = None,
        num_phoneme_tokens: int | None = None,
        dim_phoneme_emb=512,
        dim=512,
        depth=10,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_qk_norm=True,
        ff_dropout=0,
        conv_pos_embed_kernel_size=31,
        conv_pos_embed_groups=None,
        attn_dropout=0,
        attn_flash=False,
        p_drop_prob=0.2,
        frac_lengths_mask: Tuple[float, float] = (0.1, 1.),
        aligner_kwargs: dict = dict(dim_in = 80, attn_channels = 80)
    ):
        super().__init__(audio_enc_dec=audio_enc_dec, tokenizer=tokenizer, num_phoneme_tokens=num_phoneme_tokens, dim_phoneme_emb=dim_phoneme_emb, dim=dim, depth=depth, dim_head=dim_head, heads=heads, ff_mult=ff_mult, attn_qk_norm=attn_qk_norm, ff_dropout=ff_dropout, conv_pos_embed_kernel_size=conv_pos_embed_kernel_size, conv_pos_embed_groups=conv_pos_embed_groups, attn_dropout=attn_dropout, attn_flash=attn_flash, p_drop_prob=p_drop_prob, frac_lengths_mask=frac_lengths_mask, aligner_kwargs=aligner_kwargs)
        self.aligner = AlignerNet(dim_hidden=dim_phoneme_emb, **aligner_kwargs)

    @beartype
    def forward(
        self,
        *,
        cond,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_drop_prob = 0.,
        target = None,
        cond_mask = None,
        mel = None,
        phoneme_len = None,
        mel_len = None,
        phoneme_mask = None,
        mel_mask = None,
        self_attn_mask = None,
        return_aligned_phoneme_ids = False
    ):
        batch, seq_len, cond_dim = cond.shape

        cond = self.proj_in(cond)

        # text to phonemes, if tokenizer is given

        if not exists(phoneme_ids):
            assert exists(self.tokenizer)
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts)

        # construct mask if not given

        if not exists(cond_mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        cond = cond * rearrange(~cond_mask, '... -> ... 1')

        # classifier free guidance

        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, cond.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

        # phoneme id of -1 is padding

        if not exists(self_attn_mask):
            self_attn_mask = phoneme_ids != -1

        phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # force condition to be same length as input phonemes

        cond = curtail_or_pad(cond, phoneme_ids.shape[-1])

        # combine audio, phoneme, conditioning

        embed = torch.cat((phoneme_emb, cond), dim = -1)
        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        x = self.transformer(
            x,
            mask = self_attn_mask
        )

        durations = self.to_pred(x)

        if not self.training:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)]), 'need to pass phoneme_len, mel_len, phoneme_mask, mel_mask, to train duration predictor module'

        alignment_hard, _, alignment_logprob, _ = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
        target = alignment_hard

        loss_mask = cond_mask & self_attn_mask

        loss = F.l1_loss(x, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()

        #aligner loss

        align_loss = self.align_loss(alignment_logprob, phoneme_len, mel_len)
        loss = loss + align_loss

        return loss


class VoiceBox(_VB):
    """ Nothing to fix currently. Add some docs for `self.forward()` parameters.
    """
    def forward(
        self,
        x,
        *,
        times,
        cond_token_ids,
        self_attn_mask=None,
        cond_drop_prob=0.1,
        target=None,
        cond=None,
        cond_mask=None
    ):
        """
        Parameters:
            x: x_t
            times: t
            cond_token_ids: y (expended phonemes)
                Phonemes should have already expended, or else treated as SemanticTokens (w2v-bert/hubert tokens) and interpolate
            cond: x (target spectrogram)
            cond_mask
        """
        return super().forward(x, times=times, cond_token_ids=cond_token_ids, self_attn_mask=self_attn_mask, cond_drop_prob=cond_drop_prob, target=target, cond=cond, cond_mask=cond_mask)
    

class ConditionalFlowMatcherWrapper(_CFMWrapper):
    """ Deal with `self.forward()` duration prediction and aligner.
    """
    def forward(self, x1, *, mask=None, semantic_token_ids=None, phoneme_ids=None, cond=None, cond_mask=None, input_sampling_rate=None):
        """TODO: Deal with phoneme duration alignment and expansion"""
        return super().forward(x1, mask=mask, semantic_token_ids=semantic_token_ids, phoneme_ids=phoneme_ids, cond=cond, cond_mask=cond_mask, input_sampling_rate=input_sampling_rate)