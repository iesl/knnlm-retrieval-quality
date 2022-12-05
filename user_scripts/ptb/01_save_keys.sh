CHECKPOINT=ckpt/auto-ptb-run-4/checkpoint_best.pt
DSTORE_DIR=./work_data/ptb.train
MAX_TOK=512
MAX_WIN=256
DSTORE_SIZE=1001735

mkdir -p $DSTORE_DIR

python eval_lm.py data-bin/ptb \
    --path $CHECKPOINT \
    --sample-break-mode none --max-tokens $MAX_TOK \
    --softmax-batch 1024 --gen-subset train \
    --context-window $MAX_WIN --tokens-per-sample $MAX_WIN \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16


