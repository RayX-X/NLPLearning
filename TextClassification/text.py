# -*- coding:utf-8 -*-
# @Time    : 2019/10/14 16:20
# @Author  : Ray.X
import re
import zipfile  # 解压zip
import lxml.etree  # 解析xml文档
from nltk.corpus import stopwords
from nltk.tokenize.stanford import StanfordTokenizer

colors = [
    '#%dr%dr%d' % (int(2.55 * r), 150, 150) for r in clu_labels
]