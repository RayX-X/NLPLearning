# -*- coding:utf-8 -*-
# @Time    : 2019/11/21 15:52
# @Author  : Ray.X
from nltk.lm.preprocessing import pad_both_ends
from nltk import ngrams
import re
import joblib
# 读原始数据
datalist = open(r"../Data/CGEC/data.train", encoding='UTF-8').readlines()
dataline = [datalist[i].replace('\n', '').split('\t') for i in range(len(datalist))]
del datalist
# 处理成(srcSent, tgtSent)对集合
newdata = []
for i in range(len(dataline)):
    if dataline[i][1] == '0':
        newdata.append([dataline[i][2], dataline[i][2]])
    else:
        for j in range(3, len(dataline[i])):
            newdata.append([dataline[i][2], dataline[i][j]])
del dataline
# In[] (srcSent, tgtSent)过滤
from nltk.lm.preprocessing import pad_both_ends
from nltk import ngrams
import re
import joblib
panti_gram = joblib.load(r'D:\C\NLP\CGEC\panti_gram.pkl')  # 调用语言模型
# In[]
text = '这个软件让我们什么有趣的事都记录。这个软件讓我们能把任何有趣的事都记录下來。	这个软件能让我们把有趣的事都记录下来。	这个软件能让我们把任何有趣的事都记录。'
sent_list = text.split('。')
sent_list = [re.sub(r'[^\w\s]', '', sent.strip()) for sent in sent_list]
sent_list = [' '.join(w).split(' ') for w in sent_list]
while [""] in sent_list:
    sent_list.remove([""])

perplexity = [panti_gram.perplexity(sent) for sent in sent_list]
