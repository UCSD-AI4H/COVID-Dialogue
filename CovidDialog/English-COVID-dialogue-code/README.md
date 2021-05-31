# covid-dialog-english

## Preperation
* Package Installation
```
pip install cleantext
pip install fairseq==0.9.0
pip install transformers==3.5.0
pip install fire
pip install hydra-core
```

## Download Data
```
python download/download_dialogs.py
```

## BART
### Train
```
python bart_finetune.py [--n_epochs xxx]
```
* training log path: ```bart_5epochs_training_logs/```
* ```bart_5epochs_training_logs/log.txt```: evaluation losses of checkpoints.
* ```bart_5epochs_training_logs/best_model.pt```: stored best checkpoint.
* ```bart_5epochs_training_logs/generation/```: example generation after every checkpoint.

### Generation
```bash
python bart_generate.py --log_path bart_5epochs_training_logs/
```
* ```bart_5epochs_training_logs/test.hypo```: generated response.
* ```bart_5epochs_training_logs/test.gold```: ground truth.


## BART+SSL-Reg
### Train
#### Dialog generation task and SSL task are using the same BART encoder.
```
python ssl_reg_finetune.py --n_epochs 5 --weight 0.1 --text_type 0 --shared_training 'encoder'
```

#### Dialog generation task and SSL task are using the same BART decoder.
```
python ssl_reg_finetune.py --n_epochs 5 --weight 0.1 --text_type 0 --shared_training 'decoder'
```

* weight: the value of regularization parameter.
* text_type: the training sentences for SSL task.
* 	text_type=0: src_text + " " + trt_text;
*	text_type=1: src_text
*	text_type=2: trt_text
* training log path: ```bart&ssl_reg_5epochs_training_logs/```
* ```bart&ssl_reg_5epochs_training_logs/log.txt```: evaluation losses of checkpoints.
* ```bart&ssl_reg_5epochs_training_logs/best_model.pt```: stored best checkpoint.
* ```bart&ssl_reg_5epochs_training_logs/generation/```: example generation after every checkpoint.

### Generation
```bash
python bart_generate.py --log_path bart&ssl_reg_5epochs_training_logs/
```
* ```bart&ssl_reg_5epochs_training_logs/test.hypo```: generated response.
* ```bart&ssl_reg_5epochs_training_logs/test.gold```: ground truth.


## DialoGPT
### Train
```bash
python dialogpt_finetune.py --model_size [small/medium/large] --n_epochs xxx
```
Training log creation is similar to BART's.

### Generation
```bash
python dialogpt_generate.py --model_size [small/medium/large] --log_path path/to/train-log
```

## Transformer
### Binarize Data
```bash
bash transformers/bin_data.sh
```
### Train
```bash
bash transformers/train.sh
```
### Generation
```bash
bash transformers/generate.sh
```

## Evaluation
You may need to install the following perl modules (e.g. by ```cpan install```): XML:Twig, Sort:Naturally and String:Util.

We used the public code to compute the NLP evaluation metrics ```evaluation/metrics.py```

```bash
bash download/download_nist_meteor.sh

python evaluate.py \
    --hypo_path bart_5epochs_training_logs/test.hypo \
    --gold_path bart_5epochs_training_logs/test.gold
```