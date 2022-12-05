CHECKPOINT=ckpt/auto-ptb-run-4/checkpoint_best.pt
DSTORE_DIR=./work_data/ptb.train
DSTORE_SIZE=1003610
EVAL_DSTORE_DIR=./work_data/ptb.valid
EVAL_DSTORE_SIZE=42355
MAX_TOK=512
MAX_WIN=256

mkdir -p $EVAL_DSTORE_DIR

python eval_lm.py data-bin/ptb \
    --path $CHECKPOINT \
    --sample-break-mode complete --max-tokens $MAX_TOK \
    --softmax-batch 1024 --gen-subset valid \
    --context-window $MAX_WIN \
    --no-min-context  \
    --dstore-mmap $EVAL_DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size $EVAL_DSTORE_SIZE --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16


