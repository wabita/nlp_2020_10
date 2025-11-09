#学習（学習用の語彙を作る）
subword-nmt learn-bpe -s 16000 < data/tok/kyoto-train.ja > bpe.ja
subword-nmt learn-bpe -s 16000 < data/tok/kyoto-train.en > bpe.en

# サブワード化　日本語側
subword-nmt apply-bpe -c bpe.ja < data/tok/kyoto-train.ja > data/bpe/kyoto-train.ja
subword-nmt apply-bpe -c bpe.ja < data/tok/kyoto-dev.ja > data/bpe/kyoto-dev.ja
subword-nmt apply-bpe -c bpe.ja < data/tok/kyoto-test.ja > data/bpe/kyoto-test.ja
# サブワード化 英語側
subword-nmt apply-bpe -c bpe.en < data/tok/kyoto-train.en > data/bpe/kyoto-train.en
subword-nmt apply-bpe -c bpe.en < data/tok/kyoto-dev.en > data/bpe/kyoto-dev.en
subword-nmt apply-bpe -c bpe.en < data/tok/kyoto-test.en > data/bpe/kyoto-test.en

#91
fairseq-preprocess \
  --source-lang ja --target-lang en \
  --trainpref data/bpe/kyoto-train \
  --validpref data/bpe/kyoto-dev \
  --testpref data/bpe/kyoto-test \
  --destdir data-bin/my_translation_data_bpe
# "日本語->英語"
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/my_translation_data_bpe \
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
  --save-dir checkpoints/my_transformer_ja2en_bpe \
  --tensorboard-logdir tensorboard_ja2en_bpe \
  --reset-optimizer --reset-dataloader --reset-meters

#93
fairseq-generate data-bin/my_translation_data_bpe \
   --path checkpoints/my_transformer_ja2en_bpe/checkpoint_best.pt \
   --batch-size 1 \
   --beam 5 \
   --remove-bpe| tee pro/95/generate_bpe.log \

grep "BLEU4" pro/95/generate_bpe.log
#Generate test with beam=5: BLEU4 = 21.12, 52.9/26.2/15.3/9.7 (BP=0.991, ratio=0.991, syslen=26501, reflen=26735)

#95
for beam in {1..10}
do
  fairseq-generate data-bin/my_translation_data_bpe \
   --path checkpoints/my_transformer_ja2en_bpe/checkpoint_best.pt \
   --batch-size 1 \
   --beam ${beam} \
   --remove-bpe| tee pro/95/beam/beam_${beam}.log 

  #Generate test with beam=5: BLEU4 = 20.13, 50.7/24.9/14.4/9.1 (BP=1.000, ratio=1.008, syslen=26961, reflen=26734)
  grep "BLEU4" pro/95/beam/beam_${beam}.log | \
   awk '{print $7}' | tr -d ',' >> pro/95/beam.txt

done