from lhotse.recipes.libritts import download_libritts, download_librittsr, prepare_libritts, prepare_librittsr

download_librittsr("../../datasets/")
prepare_librittsr("../../datasets/LibriTTS_R", output_dir="../../datasets/LibriTTS_R_out")