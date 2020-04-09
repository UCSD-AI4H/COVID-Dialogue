# COVID Dialogue Generate Model

## Introduction

This is a pytorch implementation of Chinese COVID Dialogue Generate Model.

# Environment

The code is based on python 3.7.3 and pytorch 1.4.0 version. The code is developed and ran using one GeForce RTX 2080Ti.

## Run

In training:

`LOAD_DIR` is the directory that you store your trained model weight

`DECODER_PATH` is the file path of the trained model weight

Before running the code, download the pre-trained [encoder](https://drive.google.com/file/d/13GnYf6pj0wD7XNWrazMnoQXgUki4Tybp/view?usp=sharing) and [decoder](https://drive.google.com/file/d/1qaAUCV2alrYVrSUyM2PCmYdRuAip_DSP/view?usp=sharing) model weight.

> You can also directly download the final model [here](https://drive.google.com/file/d/1079k1JTmGQC3Qvzzru1VNiAiIkjQ7AG1/view?usp=sharing)



1. First preprocess the dataset and split the data to train, validate and test sets:

   ```shell
   $ python preprocess.py
   ```

2. Train the dialogue generate model:

   ``` shell
   $ python train.py --load_dir ${LOAD_DIR}
   ```

3. Evaluation on the trained dialogue generate model:

   ```shell
   $ python calculate_perplexity.py --decoder_path ${DECODER_PATH}
   ```

4. Generate the reply sample on the trained dialogue generate model:

   ```shell
   $ python sample_generate.py --decoder_path ${DECODER_PATH}
   ```



