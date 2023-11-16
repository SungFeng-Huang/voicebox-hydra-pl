import os
import logging
from typing import Any, Dict, Optional, Tuple, Union
from math import ceil

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import lhotse
from lhotse.recipes.utils import manifests_exist, Pathlike
from src.data.components.lhotse.tts_dataset import LhotseTextToSpeechDataset
from nemo.collections.asr.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.parts.utils.helpers import (
    clip_grad_value_,
    g2p_backward_compatible_support,
    plot_spectrogram_to_numpy,
    slice_segments,
)
from nemo.utils import model_utils
from nemo.core import ModelPT
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

log = logging.getLogger(__name__)

class LibriHeavyDataModule(ModelPT, LightningDataModule):
    """`LightningDataModule` for the LibriHeavy dataset.

    https://github.com/k2-fsa/libriheavy/blob/master/run.sh

    A `LightningDataModule` implements 7 key methods.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        cfg: DictConfig,
        corpus_dir: str = "/datasets/LibriLight/",
        manifests_dir: Pathlike = "data/LibriHeavy",
        tokenizer: BaseTokenizer = None,
        train_ds: DictConfig = None,
        validation_ds: DictConfig = None,
        test_ds: DictConfig = None,
    ) -> None:
        """Initialize a `LibriHeavyDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(cfg=cfg, trainer=None)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # NeMo does not separate LightningDataModule functions from LightningModule, therefore NeMo takes a 'cfg' arg for wrapping all hparams.
        # However, we use separated datamodule with a corresponding config file, therefore not needed to wrap the configs within 'cfg' arg.
        # This line wraps a 'cfg' as an attribute and updates self.hparams as a hyperparameter.
        self.cfg = dict(self.hparams)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # Adapted functions
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        os.makedirs(self._cfg.manifests_dir, exist_ok=True)
        for subset in self.subsets:
            if not manifests_exist(subset, self._cfg.manifests_dir, ["cuts"], "librilight"):
                log.info(f"Downloading {subset} subset.")
                os.system(f"wget -P {self._cfg.manifests_dir} -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_{subset}.jsonl.gz")
            else:
                log.info(f"Skipping download, {subset} subset exists.")

    def _setup_dataloader_from_config(self, config: Optional[Dict]) -> DataLoader[Any]:
        """Modified from https://github.com/pzelasko/NeMo/blob/feature/lhotse-integration/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L129
        """
        assert config.get("use_lhotse")

        # Note:
        #    Lhotse Dataset only maps CutSet -> batch of tensors, but does not actually
        #    contain any data or meta-data; it is passed to it by a Lhotse sampler for
        #    each sampler mini-batch.
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=LhotseTextToSpeechDataset(
                tokenizer=self.tokenizer, noise_cuts=config.get("lhotse", {}).get("noise_cuts")
            ),
        )

    # From `nemo.collections.tts.models.vits.py`
    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg and cfg.g2p is not None:
            # for backward compatibility
            if (
                self._is_model_being_restored()
                and (cfg.g2p.get('_target_', None) is not None)
                and cfg.g2p["_target_"].startswith("nemo_text_processing.g2p")
            ):
                cfg.g2p["_target_"] = g2p_backward_compatible_support(
                    cfg.g2p["_target_"]
                )

            g2p_kwargs = {}

            if "phoneme_dict" in cfg.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg, **text_tokenizer_kwargs)

    # Copied `nemo.collections.asr.models.EncDecRNNTModel` functions
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """Modify from `nemo.collections.asr.models.EncDecRNNTModel`.
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                log.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """Modify from `nemo.collections.asr.models.EncDecRNNTModel`.
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """Modify from `nemo.collections.asr.models.EncDecRNNTModel`.
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    # `LightningDataModule` original function placeholder
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = LibriHeavyDataModule()
