# Chinese-GPT-2

When training, run "gpt_train.py" for No MMI, and "gpt2_train.py --train_mmi" for MMI. Notice that, if you run the training file first time, please use "python gpt_train.py --raw" or "python gpt_train.py --train_mmi --raw" to tokenize.
When calculating the perplexity, run "gpt2_ppl.py".
When testing, run "gpt2_test.py" for no MMI and run "gpt2_test_mmi.py" MMI.


Notice that, both the training and testing file should be groups of dialogue text, which is split by a blank line.
For example, the train.txt should be like:
What are you doing?
Chating with you.
I see.

Hi, are you okay?
Yes, I am fine.

......

We use make_test.py to implement this.

When using pretrain, please download the pretrain model and put it under the file "dialogue_model".


Requirements:
transformers==2.1.1
pytorch==1.4.0
sklearn
tqdm
numpy
scipy==1.2.1