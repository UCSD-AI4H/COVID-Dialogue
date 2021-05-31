TEXT=data/english

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $TEXT/bins \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --bpe gpt2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 1000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --no-epoch-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --max-epoch 50