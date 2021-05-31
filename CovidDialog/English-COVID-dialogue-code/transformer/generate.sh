TEXT=data/english

fairseq-generate $TEXT/bins \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 32 \
    --beam 10 \
    --bpe gpt2 \
    --lenpen 2. \
    --remove-bpe \
    --skip-invalid-size-inputs-valid-test