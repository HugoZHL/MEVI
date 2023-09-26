SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}
export DATA_DIR="../data/nq_dpr"
python ensemble_nqdpr.py \
--mapping_file ${DATA_DIR}/rqmapping4_5.pkl \
--dir_path ${DATA_DIR} \
--ance_file hnsw256.txt \
--fine_file nci_result_rq45_top10_hn21015324.tsv \
--coarse_file nci_result_rq45_top10_coarse.tsv \
--ofile ${DATA_DIR}/ensemble_result.txt
