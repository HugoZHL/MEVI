export LABEL_LENGTH=4
export CODE_BITS=5
export KARY=$[2**${CODE_BITS}]
SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/nq_dpr"
python main.py \
    --n_gpu 8 \
    --mode train \
    --query_type gtq_doc_qg1 \
    --model_info base \
    --id_class bert_k30_c30_1 \
    --dataset nq_dpr \
    --Rdrop 0. \
    --save_top_k 2 \
    --train_batch_size 512 \
    --eval_batch_size 16 \
    --encode_batch_size 1024 \
    --kary $KARY --label_length_cutoff ${LABEL_LENGTH} --max_output_length $[2+${LABEL_LENGTH}] \
    --cluster_path ${DATA_DIR}/rqclus${LABEL_LENGTH}_${CODE_BITS}.pkl \
    --mapping_path ${DATA_DIR}/rqmapping${LABEL_LENGTH}_${CODE_BITS}.pkl \
    --data_dir ${DATA_DIR} \
    --tree_path ${DATA_DIR}/pq_tree${LABEL_LENGTH}_${CODE_BITS}_$KARY.pkl \
    --newid_dir ${DATA_DIR} \
    --ckpt_dir ../data/marco/ckpts \
    --wandb_token ${WANDB_TOKEN}
    # --embedding_path ${DATA_DIR}/docemb.bin \
    # --document_path ${DATA_DIR}/all_document \
