# from DPR
SCRIPT_DIR=$(dirname $(realpath $0))
cd $SCRIPT_DIR/../../data
mkdir -p nq_dpr
cd nq_dpr

wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
