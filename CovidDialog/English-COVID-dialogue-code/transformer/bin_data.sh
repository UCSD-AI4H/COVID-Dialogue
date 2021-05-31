# Preprocess/binarize the data
TEXT=data/english
fairseq-preprocess --source-lang chat_history --target-lang response \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $TEXT/bins \
    --workers 10