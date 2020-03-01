# [WIP]
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer

docs = []
file_path_list = glob2.glob("./data/*")

for file_path in file_path_list:
    print(f"正在處理 {file_path}...")
    name = os.path.basename(file_path)
    content = get_raw_content(file_path)
    content = get_regex_content(content)
    content = get_ner_content(content)
    docs.append(content)

# TF-IDF
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm=None)
X = vectorizer.fit_transform(docs)
feature_names = np.array(vectorizer.get_feature_names())
sorted_indexes = X.toarray().argsort()
for idx, indexes in enumerate(sorted_indexes):
    print(f"==={file_path_list[idx]}===")
    print(feature_names[indexes][:50])  # tfidf前50小 (過濾掉)
    print(feature_names[indexes][::-1][:50])  # tfidf前50大 (關鍵字)
