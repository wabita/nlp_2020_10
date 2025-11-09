#一文に対して翻訳するコマンド
fairseq-interactive data-bin/my_translation_data_ja2en \
    --path checkpoints/my_transformer_ja2en/checkpoint_best.pt \
    --beam 5 --source-lang ja --target-lang en \
    --buffer-size 1024 --batch-size 1 --remove-bpe
