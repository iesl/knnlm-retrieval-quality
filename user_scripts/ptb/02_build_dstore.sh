CHECKPOINT=ckpt/auto-ptb-run-4/checkpoint_best.pt
DSTORE_DIR=./work_data/ptb.train
MAX_TOK=512
MAX_WIN=256
DSTORE_SIZE=1003610

mkdir -p $DSTORE_DIR

python build_dstore.py \
    --write-interval 100000 \
    --dstore_mmap $DSTORE_DIR/dstore \
    --dstore_size $DSTORE_SIZE \
    --faiss_index $DSTORE_DIR/knn.index \
    --num_keys_to_add_at_a_time 50000 \
    --ncentroids 512 \
    --starting_point 0 --dstore_fp16

