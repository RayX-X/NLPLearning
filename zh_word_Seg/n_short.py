# -*- coding:utf-8 -*-
# @Time    : 2019/10/22 9:12
# @Author  : Ray.X
import pandas as pd
import numpy as np
import re
import lxml.etree
df = open('../Data/dict/baidu_dict.txt', 'r', encoding='utf-8').readlines()
word_dict = []
for line in df:
    word_dict.append(line.split()[1])

doc = lxml.etree.parse(open('../Data/ted_zh-cn-20160408.xml', 'r', encoding='utf-8'))
content = (doc.xpath('//content/text()'))   # 获取<content>下的文本 数组
del doc

test_list = [(re.sub(r'[^\w\s]', '', str(content[i]))).split() for i in range(len(content))]
