#学習率とバッチサイズの組み合わせ最適を求める
RESULT_DIR="pro/97/bleu"
RESULT_FILE="${RESULT_DIR}/bleu.txt"
mkdir -p ${RESULT_DIR}

# 古い結果ファイルを削除し、ヘッダーを書き込む
rm -f ${RESULT_FILE}
echo "BLEU   LRS    TOKENS" > ${RESULT_FILE}

best_bleu=0
best_setting=""

for LRS in 1e-3 5e-4 3e-4; do
    for TOKENS in 1024 2048 4096; do
        SAVE_DIR=checkpoints/my_transformer_lr${LRS}_tok${TOKENS}
        LOG_FILE=${RESULT_DIR}/lr${LRS}_tok${TOKENS}.log

        echo "===== Training: LR=${LRS}, TOKENS=${TOKENS} ====="

        #学習 (この部分はOK)
        CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/my_translation_data_bpe \
            --arch transformer_iwslt_de_en \
            --share-decoder-input-output-embed \
            --source-lang ja --target-lang en \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
            --lr ${LRS} --lr-scheduler inverse_sqrt --warmup-updates 4000 \
            --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
            --dropout 0.3 --weight-decay 0.0001 \
            --max-tokens ${TOKENS} \
            --max-epoch 10 \
            --save-interval 5 \
            --amp \
            --no-progress-bar --log-interval 100 \
            --save-dir ${SAVE_DIR} \
            --tensorboard-logdir tensorboard_ja2en_bpe \
            --reset-optimizer --reset-dataloader --reset-meters
        
        echo "🧪 Generating translations..."
        #評価 (--- 修正箇所 1 ---)
        fairseq-generate data-bin/my_translation_data_bpe \
            --path ${SAVE_DIR}/checkpoint_best.pt \
            --batch-size 64 \
            --beam 7 \
            --remove-bpe \
            --arch transformer_iwslt_de_en \
            --share-decoder-input-output-embed \
            --dropout 0.3 \
            --amp \
            | tee ${LOG_FILE}

        # --- 修正箇所 2: BLEUスコアの抽出と保存 ---
        BLEU=$(grep "BLEU4" ${LOG_FILE} | awk '{print $7}' | tr -d ',')

        if [[ -z "${BLEU}" ]]; then
            echo "⚠️ No BLEU score found for LR=${LRS}, TOKENS=${TOKENS}"
            BLEU=0
        fi

        # BLEUスコアとパラメータを一緒にファイルへ書き込む
        echo "${BLEU} ${LRS} ${TOKENS}" >> ${RESULT_FILE}
        echo "BLEU for LR=${LRS}, TOKENS=${TOKENS} = ${BLEU}"

        # 実行中にベストスコアを追跡
        if (( $(echo "${BLEU} > ${best_bleu}" | bc -l) )); then
            best_bleu=${BLEU}
            best_setting="LR=${LRS}, TOKENS=${TOKENS} (BLEU=${BLEU})"
        fi

    done
done

echo "completed."
echo "Best setting: ${best_setting}"

echo ""
echo "--- Final Results (Top 5 sorted by BLEU) ---"
tail -n +2 ${RESULT_FILE} | sort -nr -k1 | head -n 5

