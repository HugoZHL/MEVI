SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/marco"
python main.py \
--n_gpu 8 \
--mode eval \
--query_type gtq --co_neg_from clus \
--model_info base \
--id_class bert_k30_c30_1 \
--dataset marco \
--Rdrop 0. \
--eval_batch_size 2 \
--encode_batch_size 1024 \
--document_encoder ${DOCUMENT_ENCODER} \
--recall_level both \
--qtower encmask_dec --query_embed_accum attenpool \
--pq_loss ce \
--pq_type rq \
--codebook 1 --subvector_num 4 --subvector_bits 5 \
--use_gumbel_softmax 0 --pq_softmax_tau 1 --pq_hard_softmax_topk 1 \
--pq_negative none \
--fixnci --fixpq \
--document_encoder_from_pretrained 0 --not_load_document_encoder 0 \
--no_nci_loss 1 --query_encoder twin \
--num_return_sequences 10 \
--save_hard_neg 8841823 \
--pq_path ${DATA_DIR}/ance/rqcodebook4_5.pt \
--pq_cluster_path ${DATA_DIR}/ance/rqclus4_5.pkl \
--nci_ckpt ${DATA_DIR}/ckpts/nci_marco_gtq_doc_qg10_base_k32_dem2_ada1_1_4_rdrop0.1_0.0_0_cut4_ance_lre2.0d1.0_epoch=26-recall100=0.886079.ckpt \
--data_dir ${DATA_DIR}/origin \
--newid_dir ${DATA_DIR}/ance \
--document_path ${DATA_DIR}/${DOCUMENT_ENCODER}/all_document \
--ckpt_dir ${DATA_DIR}/ckpts \
--embedding_path ${DATA_DIR}/${DOCUMENT_ENCODER}/docemb.bin \
--custom_save_path ${DATA_DIR}/${DOCUMENT_ENCODER}/nci_result_rq45_top10.tsv
