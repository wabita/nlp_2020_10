for beam in {1..10}
do
  fairseq-generate data-bin/my_translation_data_ja2en \
   --path checkpoints/my_transformer_ja2en/checkpoint_best.pt \
   --batch-size 1 \
   --beam ${beam} \
   --remove-bpe| tee pro/94/beam/beam_${beam}.log 

  #Generate test with beam=5: BLEU4 = 20.13, 50.7/24.9/14.4/9.1 (BP=1.000, ratio=1.008, syslen=26961, reflen=26734)
  grep "BLEU4" pro/94/beam/beam_${beam}.log | \
   awk '{print $7}' | tr -d ',' >> pro/94/beam.txt

done