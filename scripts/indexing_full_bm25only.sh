#!/usr/bin/env bash

# ./indexing_full_bm25only.sh k1 b
#
# Zhuyun use follwing setting:
# 	Baseline BM25: k1=0.6, b=0.8
# 	DeepCT: k1=10, b=0.9
# 	DeepCT-sqrt: k1=18, b=0.7


##########  Output  ############
# MRR @10: 0.19059916086778556
# QueriesRanked: 6980
#################################


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


# get passage data
mkdir collections/msmarco-passage
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage

# Alternative mirror:
# wget https://www.dropbox.com/s/9f54jg2f71ray3b/collectionandqueries.tar.gz -P collections/msmarco-passage

tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C collections/msmarco-passage


# tsv to json
python tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path collections/msmarco-passage/collection.tsv \
 --output-folder collections/msmarco-passage/collection_jsonl

# indexing 
sh target/appassembler/bin/IndexCollection -threads 9 -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -input collections/msmarco-passage/collection_jsonl \
 -index indexes/msmarco-passage/lucene-index-msmarco -storePositions -storeDocvectors -storeRaw 

# Performing Retrieval on the Dev Queries
python tools/scripts/msmarco/filter_queries.py \
--qrels collections/msmarco-passage/qrels.dev.small.tsv \
--queries collections/msmarco-passage/queries.dev.tsv \
--output collections/msmarco-passage/queries.dev.small.tsv


sh target/appassembler/bin/SearchMsmarco -k1 $1 -b $2 -hits 1000 -threads 1 \
 -index indexes/msmarco-passage/lucene-index-msmarco \
 -queries collections/msmarco-passage/queries.dev.small.tsv \
 -output runs/run.msmarco-passage.dev.small.tsv


# Evaluate
python tools/scripts/msmarco/msmarco_eval.py \
 collections/msmarco-passage/qrels.dev.small.tsv runs/run.msmarco-passage.dev.small.tsv


cd ..
rm -rf anserini
echo "deleted anserini and finished!"
