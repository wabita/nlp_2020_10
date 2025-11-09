pip install fairseq==0.12.2

fairseq-preprocess \
    --source-lang ja --target-lang en \
    --trainpref data/tok/kyoto-train \
    --validpref data/tok/kyoto-dev \
    --testpref data/tok/kyoto-test \
    --destdir data-bin/my_translation_data \
    --workers 4

# "日本語->英語"
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/my_translation_data_ja2en \
  --arch transformer_iwslt_de_en \
  --share-decoder-input-output-embed \
  --source-lang ja --target-lang en \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --dropout 0.3 --weight-decay 0.0001 \
  --max-tokens 4096 \
  --max-epoch 10 \
  --save-interval 5 \
  --amp \
  --no-progress-bar --log-interval 100 \
  --save-dir checkpoints/my_transformer_ja2en \
  --tensorboard-logdir tensorboard_ja2en \
  --reset-optimizer --reset-dataloader --reset-meters
  
# "英語->日本語"
# CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/my_translation_data \
#   --arch transformer \
#   --source-lang en --target-lang ja \
#   --optimizer adam --lr 0.0005 --batch-size 16 \
#   --max-tokens 2048 \
#   --lr-scheduler inverse_sqrt \
#   --warmup-updates 2000 \
#   --max-epoch 10 \
#   --dropout 0.1 \
#   --save-dir checkpoints/my_transformer_single


