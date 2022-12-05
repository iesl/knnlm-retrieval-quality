INP_DIR=data/ptb
OUT_DIR=data-bin/ptb

python preprocess.py \
    --only-source \
    --srcdict $INP_DIR/dict.txt \
    --trainpref $INP_DIR/train.txt \
    --validpref $INP_DIR/dev.txt \
    --testpref $INP_DIR/test.txt \
    --destdir $OUT_DIR \
    --workers 20

