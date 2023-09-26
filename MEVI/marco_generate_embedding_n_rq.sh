SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/marco"
python main.py \
--n_gpu 8 \
--mode train \
--query_type gtq --co_neg_from clus \
--model_info base \
--id_class bert_k30_c30_1 \
--dataset marco \
--encode_batch_size 1024 \
--recall_level fine \
--pq_type rq \
--only_gen_rq 1 \
--codebook 1 --subvector_num 4 --subvector_bits 5 \
--document_encoder_from_pretrained 0 --not_load_document_encoder 0 \
--no_nci_loss 1 --query_encoder twin \
--num_return_sequences 10 \
--document_encoder ${DOCUMENT_ENCODER} \
--pq_path ${DATA_DIR}/${DOCUMENT_ENCODER}/rqcodebook4_5.pt \
--pq_cluster_path ${DATA_DIR}/${DOCUMENT_ENCODER}/rqclus4_5.pkl \
--data_dir ${DATA_DIR}/origin \
--ckpt_dir ${DATA_DIR}/ckpts \
--newid_dir ${DATA_DIR}/${DOCUMENT_ENCODER} \
--document_path ${DATA_DIR}/${DOCUMENT_ENCODER}/all_document \
--embedding_path ${DATA_DIR}/${DOCUMENT_ENCODER}/docemb.bin
