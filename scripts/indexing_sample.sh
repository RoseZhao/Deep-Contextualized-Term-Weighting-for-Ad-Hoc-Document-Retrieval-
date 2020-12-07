#!/usr/bin/env bash

# ./indexing_sample.sh sample_collection_folder_path sample_qrels_path k1 b
#
# Zhuyun use follwing setting:
# 	Baseline BM25: k1=0.6, b=0.8
# 	DeepCT: k1=10, b=0.9
# 	DeepCT-sqrt: k1=18, b=0.7


# install and setup
git clone https://github.com/castorini/anserini.git --recurse-submodules
pip install pyserini==0.9.4.0
cd anserini/

cd tools/eval
tar xvfz trec_eval.9.0.4.tar.gz
cd trec_eval.9.0.4
make
cd ../../..
cd tools/eval/ndeval
make
cd ../../..

mkdir collections/msmarco-passage
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage
tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C collections/msmarco-passage
sudo apt-get install maven
mvn clean package appassembler:assemble


# indexing 
sh target/appassembler/bin/IndexCollection -threads 9 -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -input ../$1 \
 -index indexes/msmarco-passage/sample_100_index -storePositions -storeDocvectors -storeRaw 


# Filter queries on sampeled qrels set
python tools/scripts/msmarco/filter_queries.py \
--qrels ../$2 \
--queries collections/msmarco-passage/queries.dev.tsv \
--output collections/msmarco-passage/queries.dev.small.tsv


# Performing Retrieval on the Dev Queries
sh target/appassembler/bin/SearchMsmarco -k1 $3 -b $4 -hits 1000 -threads 1 \
 -index indexes/msmarco-passage/sample_100_index \
 -queries collections/msmarco-passage/queries.dev.small.tsv \
 -output runs/run.msmarco-passage.dev.small.tsv


# Evaluate
python tools/scripts/msmarco/msmarco_eval.py \
 ../$2 runs/run.msmarco-passage.dev.small.tsv


cd ..
rm -rf anserini
echo "deleted anserini and finished!"
