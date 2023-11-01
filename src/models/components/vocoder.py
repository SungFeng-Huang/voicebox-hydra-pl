from voicebox_pytorch.voicebox_pytorch import AudioEncoderDecoder, MelVoco, EncodecVoco

class MelVocoder(MelVoco):
    @property
    def downsample_factor(self):
        return 1.
    
EncodecVocoder = EncodecVoco