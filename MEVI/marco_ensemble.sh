SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/marco"
python ensemble_marco.py \
--mapping_file ${DATA_DIR}/ance/rqmapping4_5.pkl \
--gt_file ${DATA_DIR}/origin/dev_mevi_dedup.tsv \
--ance_file ${DATA_DIR}/${DOCUMENT_ENCODER}/hnsw256.txt \
--coarse_file ${DATA_DIR}/${DOCUMENT_ENCODER}/nci_result_rq45_top10_coarse.tsv \
--fine_file ${DATA_DIR}/${DOCUMENT_ENCODER}/nci_result_rq45_top10_hn8841823.tsv \
--ofile ${DATA_DIR}/${DOCUMENT_ENCODER}/ensemble_result.txt
