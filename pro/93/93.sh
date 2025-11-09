fairseq-generate data-bin/my_translation_data_ja2en \
   --path checkpoints/my_transformer_ja2en/checkpoint_best.pt \
   --batch-size 1 \
   --beam 5 \
   --remove-bpe| tee pro/93/generate_ja2en.log \

grep "BLEU4" pro/93/generate_ja2en.log
#Generate test with beam=5: BLEU4 = 20.13, 50.7/24.9/14.4/9.1 (BP=1.000, ratio=1.008, syslen=26961, reflen=26734)