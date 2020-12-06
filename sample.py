import csv
import random

csv.field_size_limit(100000000)
passages = set()
with open("collection.tsv", encoding='utf-8', mode="r") as f:
    rows = csv.reader(f, delimiter="\t")
    for i, row in enumerate(rows):
        qid = row[0]
        passages.add(qid)
print("total passage num: {}".format(len(passages)))

sample_passages = set(random.sample(passages, len(passages) // 10))

with open("collection.10%.tsv", encoding='utf-8', mode="w") as f1:
    with open("collection.tsv", encoding='utf-8', mode="r") as f:
        rows = csv.reader(f, delimiter="\t")
        count = 0
        for i, row in enumerate(rows):
            qid = row[0]
            if qid in sample_passages:
                f1.write("\t".join(row) + "\n")
                count += 1             
print("sample passage num: {}".format(count))

with open("qrels.dev.10%.tsv", encoding="utf-8", mode="w") as f1:
    with open("qrels.dev.tsv", encoding='utf-8', mode="r") as f:
        rows = csv.reader(f, delimiter="\t")
        count = 0
        for i, row in enumerate(rows):
            qid, pid = row[0], row[2]
            if pid in sample_passages:
                f1.write("\t".join(row) + "\n")
                count += 1
print("sample qrel num: {}".format(count))
