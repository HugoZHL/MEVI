# [NeurIPS 2023] Model-enhanced Vector Index ([Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ac112e8ffc4e5b9ece32070440a8ca43-Paper-Conference.pdf))

## Environment

[Option 1] Create conda environment:

```bash
conda env create -f environment.yml
conda activate mevi
```

[Option 2] Use docker:

```bash
docker pull hugozhl/nci:latest
```

## MSMARCO Passage

### Data Process

[1] Download and preprocess:
```bash
bash dataprocess/msmarco_passage/download_data.sh
python dataprocess/msmarco_passage/prepare_origin.py \
	--data_dir data/marco --origin
```

[2] Tokenize documents:
```bash
# tokenize for T5-ANCE and AR2

# T5-ANCE
python dataprocess/msmarco_passage/prepare_passage_tokenized.py \
	--output_dir data/marco/ance \
	--document_path data/marco/raw/corpus.tsv \
	--dataset marco --model ance
rm data/marco/ance/all_document_indices_*.pkl
rm data/marco/ance/all_document_tokens_*.bin
rm data/marco/ance/all_document_masks_*.bin

# AR2
python dataprocess/msmarco_passage/prepare_passage_tokenized.py \
	--output_dir data/marco/ar2 \
	--document_path data/marco/raw/corpus.tsv \
	--dataset marco --model ar2
rm data/marco/ar2/all_document_indices_*.pkl
rm data/marco/ar2/all_document_tokens_*.bin
rm data/marco/ar2/all_document_masks_*.bin
```

[3] Query generation for augmentation:

We used the [docT5query checkpoint](https://huggingface.co/castorini/doc2query-t5-base-msmarco) as in NCI. The QG data is only for training.

Please download the finetuned docT5query ckpt to `data/marco/ckpts/doc2query-t5-base-msmarco`

```bash
# MUST download the finetuned docT5query ckpt before running the scripts
python dataprocess/msmarco_passage/doc2query.py --data_dir data/marco
# if the qg data has bad quality, e.g. empty query or many duplicate queries, add another script below
python dataprocess/msmarco_passage/complement_qg10.py --data_dir data/marco # Optional
```

[4] Generate document embeddings and construct RQ

For T5-ANCE, please download [T5-ANCE checkpoint](https://huggingface.co/OpenMatch/t5-ance) to `data/marco/ckpts/t5-ance`.

For AR2, please download [AR2 checkpoint](https://drive.google.com/file/d/1KHYQnleuBj7pgRsdCmLm58XtuJJArXPz/view?usp=sharing) to `data/marco/ckpts/ar2g_marco_finetune.pkl` and [coCondenser checkpoint](https://huggingface.co/Luyu/co-condenser-marco-retriever) to `data/marco/ckpts/co-condenser-marco-retriever`

```bash
# MUST download the checkpoints before running the scripts
export DOCUMENT_ENCODER=ance
# export DOCUMENT_ENCODER=ar2 # use this line for ar2
bash MEVI/marco_generate_embedding_n_rq.sh
```

### Training

Train the RQ-based NCI.
```bash
export DOCUMENT_ENCODER=ance
# export DOCUMENT_ENCODER=ar2 # use this line for ar2
export WANDB_TOKEN="your wandb token"
bash MEVI/marco_train_nci_rq.sh
```

### Twin-tower Model Evaluation

First generate query embeddings.
```bash
# for T5-ANCE
python MEVI/generate.py \
	--query_file data/marco/origin/dev_mevi_dedup.tsv \
	--model_path data/marco/ckpts/t5-ance \
	--tokenizer_path data/marco/ckpts/t5-ance \
	--query_embedding_path data/marco/ance/query_emb.bin \
	--gpus 0,1,2,3,4,5,6,7 --gen_query

# for AR2
python MEVI/generate.py \
	--query_file data/marco/origin/dev_mevi_dedup.tsv \
	--model_path data/marco/ckpts/ar2g_marco_finetune.pkl \
	--tokenizer_path bert-base-uncased \
	--query_embedding_path data/marco/ar2/query_emb.bin \
	--gpus 0,1,2,3,4,5,6,7 --gen_query
```

Then use faiss for ANN search.
```bash
# for T5-ANCE; if for AR2, change the ance directory to ar2 directory
python MEVI/faiss_search.py \
	--query_path data/marco/ance/query_emb.bin \
	--doc_path data/marco/ance/docemb.bin \
	--output_path data/marco/ance/hnsw256.txt \
	--raw_query_path data/marco/origin/dev_mevi_dedup.tsv \
	--param HNSW256
```


### Sequence-to-sequence Model Evaluation

Please download our [checkpoint for MSMARCO Passage](https://drive.google.com/file/d/1OjCw6Q1iAgUr2GByDuXe541vr0WTyoaH/view?usp=sharing) or train from scratch before evaluation, and put the checkpoint in `data/marco/ckpts`. If using the downloaded checkpoint, please also download the corresponding [RQ files](https://drive.google.com/drive/folders/1xa5koU1uAOajdbuLqIXigAhFXDtb_qoQ?usp=sharing).

```bash
# MUST download or train a ckpt before running the scripts
export DOCUMENT_ENCODER=ance
# export DOCUMENT_ENCODER=ar2 # use this line for ar2
bash MEVI/marco_eval_nci_rq.sh
```

### Ensemble

Ensemble the results from the twin-tower model and the sequence-to-sequence model.

```bash
export DOCUMENT_ENCODER=ance
# export DOCUMENT_ENCODER=ar2 # use this line for ar2
bash MEVI/marco_ensemble.sh
```

## Natural Questions (DPR version)

### Data Process

[1] Download and preprocess:
```bash
bash dataprocess/NQ_dpr/download_data.sh
python dataprocess/NQ_dpr/preprocess.py --data_dir data/nq_dpr
```

[2] Tokenize documents:
```bash
# use AR2
python dataprocess/NQ_dpr/tokenize_passage_ar2.py \
	--output_dir data/nq_dpr \
	--document_path data/nq_dpr/corpus.tsv
rm data/nq_dpr/all_document_indices_*.pkl
rm data/nq_dpr/all_document_tokens_*.bin
rm data/nq_dpr/all_document_masks_*.bin
```

[3] Query generation for augmentation:

We used the docT5query checkpoint as in NCI. The QG data is only for training. Please refer to the QG section for MSMARCO Passage.

```bash
# download finetuned docT5query ckpt to data/marco/ckpts/doc2query-t5-base-msmarco
python dataprocess/NQ_dpr/doc2query.py \
	--data_dir data/nq_dpr --n_gen_query 1 \
	--ckpt_path data/marco/ckpts/doc2query-t5-base-msmarco
```

[4] Generate document embeddings and construct RQ

Please download [AR2 checkpoint](https://drive.google.com/file/d/1SV5UPieyLBAHzk9Fujrudc3h-er2WMZv/view?usp=sharing) to `data/marco/ckpts/ar2g_nq_finetune.pkl` and [ERNIE checkpoint](https://huggingface.co/PaddlePaddle/ernie-2.0-base-en) to `data/marco/ckpts/ernie-2.0-base-en`

```bash
# MUST download the checkpoints before running the scripts
bash MEVI/nqdpr_generate_embedding_n_rq.sh
```

[5] Tokenize query

Since NQ has too many augmented queries, to eliminate runtime memory usage, we tokenize query to enable memmap.
```bash
python dataprocess/NQ_dpr/tokenize_query.py \
	--output_dir data/nq_dpr \
	--tok_train 1 --tok_corpus 1 --tok_qg 1
```

[6] Get answers

We sort the answers for fast evaluation. (Time-consuming! Please download the [processed binary files](https://drive.google.com/drive/folders/1wyg8FFGgEtRrbKEf-m9YmEVlcZChCZW9?usp=sharing) if necessary.)
```bash
python dataprocess/NQ_dpr/get_answers.py \
	--data_dir data/nq_dpr \
	--dev 1 --test 1
python dataprocess/NQ_dpr/get_inverse_answers.py \
	--data_dir data/nq_dpr \
	--dev 1 --test 1
```

### Training

Train the RQ-based NCI.

```bash
export WANDB_TOKEN="your wandb token"
bash MEVI/nqdpr_train_nci_rq.sh
```


### Twin-tower Model Evaluation

First generate query embeddings.
```bash
python MEVI/generate.py \
	--query_file data/nq_dpr/nq-test.qa.csv \
	--model_path data/marco/ckpts/ar2g_nq_finetune.pkl \
	--tokenizer_path bert-base-uncased \
	--query_embedding_path data/nq_dpr/query_emb.bin \
	--gpus 0,1,2,3,4,5,6,7 --gen_query
```

Then use faiss for ANN search.
```bash
python MEVI/faiss_search.py \
	--query_path data/nq_dpr/query_emb.bin \
	--doc_path data/nq_dpr/docemb.bin \
	--output_path data/nq_dpr/hnsw256.txt \
	--raw_query_path data/nq_dpr/nq-test.qa.csv \
	--param HNSW256
```

### Sequence-to-sequence Model Evaluation

Please download our [checkpoint for NQ](https://drive.google.com/file/d/1RjomQkJj3oDnNlaCQTfRtSqIGZlWqqdD/view?usp=sharing) or train from scratch before evaluation, and put the checkpoint in `data/marco/ckpts`. If using the downloaded checkpoint, please also download the corresponding [RQ files](https://drive.google.com/drive/folders/1wyg8FFGgEtRrbKEf-m9YmEVlcZChCZW9?usp=sharing).

```bash
# MUST download or train a ckpt before running the scripts
bash MEVI/nqdpr_eval_nci_rq.sh
```

### Ensemble

Ensemble the results from the twin-tower model and the sequence-to-sequence model.

```bash
bash MEVI/nqdpr_ensemble.sh
```

## Citation
If you find this work useful, please cite [our paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ac112e8ffc4e5b9ece32070440a8ca43-Paper-Conference.pdf).


## Acknowledgement
We learned a lot and borrowed some code from the following projects when building MEVI.

* [NCI](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)
* [Transformers](https://github.com/huggingface/transformers)
* [OpenMatch](https://github.com/openmatch/openmatch)
* [DPQ](https://github.com/facebookresearch/DPR)
