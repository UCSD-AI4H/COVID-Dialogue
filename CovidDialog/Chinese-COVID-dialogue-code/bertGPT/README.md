# Chinese-bertGPT 

1. (1)When training, first run command "python preprocess.py", then run "python bert_gpt_train.py".
you need first to download the original trained model by https://drive.google.com/file/d/17ywZ4LJNukGJNiMuGcIeSqtcBRiiJUnB/view?usp=sharing
(2) If you would like to run BERT-GPT+SSL-Reg, first download Chinese BERT model from https://github.com/ymcui/Chinese-BERT-wwm, then run "python bert_gpt_train_sslreg.py"

2. When calculate the perplexity, run "python bert_gpt_perplexity.py --decoder_path ${the path of the model you save}".

3. When testing, run "python generate.py --decoder_path ${the path of the model you save}" to get the generated dialogues file,
and run "python validate.py --file_name generate_sentences.txt" to calculate the metrics.

Requirements:
chinese-gpt==0.1.3
pytorch==1.4.0
fire
tqdm
numpy
allennlp==0.9.0
pytorch-pretrained-bert==0.6.2
nltk