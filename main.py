# [WIP]
import re
import os
import json
import glob
import glob2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from utils import (
    get_ner_content,
    get_raw_content,
    get_regex_content,
    get_sw_content,
    get_stop_words,
)

docs = []
file_path_list = glob2.glob("./data/*")[:3]

for file_path in file_path_list:
    print(f"正在處理 {file_path}...")
    name = os.path.basename(file_path)
    content = get_raw_content(file_path)  # 原始內容
    content = get_regex_content(content)  # 正則過濾
    content = get_sw_content(content)  # 去除停用詞
    content = get_ner_content(content)  # NER過濾
    docs.append(content)

# TF-IDF
vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b", norm=None, stop_words=get_stop_words()
)
X = vectorizer.fit_transform(docs)
feature_names = np.array(vectorizer.get_feature_names())
sorted_indexes = X.toarray().argsort()
# for idx, indexes in enumerate(sorted_indexes):
#     print(f"==={file_path_list[idx]}===")
#     print(feature_names[indexes][:50])  # tfidf前50小 (過濾掉)
#     print(feature_names[indexes][::-1][:50])  # tfidf前50大 (關鍵字)

output = {}
for idx, indexes in enumerate(sorted_indexes):
    key = os.path.basename(file_path_list[idx])
    print(feature_names[indexes][::-1][:50])  # tfidf前50大 (關鍵字)
    output[key] = feature_names[indexes][::-1][:50].tolist()

with open("tfidf_top50_each_doc.json", "w") as f:
    f.write(json.dumps(output))
