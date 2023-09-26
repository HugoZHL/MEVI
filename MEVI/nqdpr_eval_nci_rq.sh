SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/nq_dpr"
python main.py \
--n_gpu 8 \
--mode eval \
--query_type gtq --co_neg_from clus \
--model_info base \
--id_class bert_k30_c30_1 \
--dataset nq_dpr \
--Rdrop 0. \
--eval_batch_size 2 \
--encode_batch_size 1024 \
--document_encoder ar2 \
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
--save_hard_neg 21015324 \
--pq_path ${DATA_DIR}/rqcodebook4_5.pt \
--pq_cluster_path ${DATA_DIR}/rqclus4_5.pkl \
--nci_ckpt ${DATA_DIR}/../marco/ckpts/nci_nq_dpr_gtq_doc_qg1_base_k32_dem2_ada1_1_4_rdrop0.1_0.0_0_cut4_NQ_dpr_lre2.0d1.0_epoch=79-hitrate100=0.831619 \
--data_dir ${DATA_DIR} \
--newid_dir ${DATA_DIR} \
--document_path ${DATA_DIR}/all_document \
--ckpt_dir ../data/marco/ckpts \
--embedding_path ${DATA_DIR}/docemb.bin \
--custom_save_path ${DATA_DIR}/nci_result_rq45_top10.tsv
