# -*- coding:utf-8 -*-
# @Time    : 2019/12/16 11:15
# @Author  : Ray.X
"""
    add_one
"""
from nltk.tokenize import word_tokenize
from nltk import bigrams, FreqDist
from math import log

# 读取数据 小写 替换符号 分句
dataset = open("train_LM.txt", 'r+', encoding='utf-8').read().lower()\
    .replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ')\
    .replace(':', ' ').replace(';', ' ').replace('<', ' ').replace('>', ' ').replace('/', ' ')\
    .split("__eou__")
testset = open("test_LM.txt", 'r+', encoding='utf-8').read().lower()\
    .replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ')\
    .replace(':', ' ').replace(';', ' ').replace('<', ' ').replace('>', ' ').replace('/', ' ')\
    .split("__eou__")

unigramsDist = FreqDist()  # uni-gram词频数字典
for i in dataset:
    sWordFreq = FreqDist(word_tokenize(i))  # 每一句的词频数字典
    for j in sWordFreq:
        if j in unigramsDist:
            unigramsDist[j] += sWordFreq[j]
        else:
            unigramsDist[j] = sWordFreq[j]
# 加入未登录词
for i in testset:
    word = word_tokenize(i)  # 每一句的词频数字典
    for j in word:
        if j not in unigramsDist:
            unigramsDist[j] = 0

# 频数转化为频率  使用加一平滑法   unigramsDist.B()表示每个词都加一后的增加量

s = unigramsDist.N() + unigramsDist.B()
unigramsFreq = FreqDist()
for i in unigramsDist:
    unigramsFreq[i] = (unigramsDist[i] + 1) / s

X = sum(unigramsFreq.values())

ppt = []
for sentence in testset:
    logprob = 0
    wt = 0
    for word in word_tokenize(sentence):
        if word in unigramsFreq:
            logprob += log(unigramsFreq[word], 2)
            wt += 1
    if wt > 0:
        ppt.append([sentence, pow(2, -(logprob / wt))])

temp = 0
for i in ppt:
    temp += i[1]
print("一元语法模型的困惑度:", temp / len(ppt))

# 二元
w2gram = {}     # 可能存在的以w为开头的2-gram的种类数量
bigramsDist = FreqDist()
for sentence in dataset:
    sWordFreq = FreqDist(bigrams(word_tokenize(sentence)))
    for j in sWordFreq:
        if j in bigramsDist:
            bigramsDist[j] += sWordFreq[j]
        else:
            bigramsDist[j] = sWordFreq[j]
            if j[0] in w2gram:
                w2gram[j[0]] += 1
            else:
                w2gram[j[0]] = 1

# 加入未登录词
for sentence in testset:
    word = bigrams(word_tokenize(sentence))
    for j in word:
        if j not in bigramsDist:
            bigramsDist[j] = 0

# 频数转化为频率  使用加一平滑法   unigramsDist.B()表示每个词都加一后的增加量
bigramsFreq = FreqDist()
A = FreqDist()
for i in bigramsDist:
    bigramsFreq[i] = (bigramsDist[i] + 1) / (w2gram[i[0]] + len(unigramsDist))



ppt = []
for sentence in testset:
    logprob = 0
    wt = 0
    for word in bigrams(word_tokenize(sentence)):
        if word in bigramsFreq:
            logprob += log(bigramsFreq[word], 2)
            wt += 1
    if wt > 0:
        ppt.append([sentence, pow(2, -(logprob / wt))])\

temp = 0
for i in ppt:
    temp += i[1]
print("二元语法模型的困惑度:", temp / len(ppt))
