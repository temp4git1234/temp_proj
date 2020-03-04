import re
import os
import json
import docx
import glob2
from pathlib import Path
from ner import Parser

tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
}
p = Parser()
p.load_models("models/")


def get_stop_words():
    """
    載入停用詞
    """
    with open("stopwords-en.txt", "r", encoding="utf8") as f:
        stop_words = f.readlines()
        stop_words = [sw.replace("\n", "") for sw in stop_words]
    return stop_words


def get_raw_content(file_path):
    """
    獲取檔案內文
    input: 檔案路徑
    output: 檔案內文
    """
    if Path(file_path).suffix == ".docx":
        paragraphs = docx.Document(file_path).paragraphs
        return " ".join([paragraph.text for paragraph in paragraphs])

    elif Path(file_path).suffix in [".csv", ".txt"]:
        for encoding in ["utf8", "big5", "latin-1"]:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()
                return content
            except:
                pass

    else:
        raise NotImplementedError


def get_sw_content(content):
    """
    去除停用詞
    """
    stop_words = get_stop_words()
    content = " ".join(
        list(filter(lambda word: word not in stop_words, content.split(" ")))
    )
    return content


def get_regex_content(content):
    """
    依照需求(regex)過濾檔案內文
    input: 原始的檔案內文
    output: 過濾後的檔案內文
    """
    content = re.sub(
        "\s+", " ", content
    )  # 避免換行符去除後 造成兩詞相連 (不保留換行等符號 但需保留詞間空格)
    content = re.sub("[^\w| ]", "", content)  # 只保留空格、英數字
    while re.search(" " * 2, content):  # 詞間最多只保留一個空格
        content = re.sub("  ", "", content)
    return content


def get_ner_content(content):
    """
    根據 ner 過濾人名
    """
    return " ".join([tp[0] for tp in p.predict(content) if not "PER" in tp[1]])
