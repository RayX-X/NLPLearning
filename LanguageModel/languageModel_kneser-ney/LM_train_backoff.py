#!/usr/bin/python
#coding = utf-8

"""
    This file achieves to train the language model using the back off Kneser Ney. It can divide the ngrams, calculate the parameter D, calculate
    the probabilities and back weight, create the language model file.
"""

import sys
import re
from collections import Counter
from itertools import combinations
from math import log
import io

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')

traintext = sys.argv[2]
lmlist = sys.argv[1]
print('---opening files---')
# wordsobject = open(traintext, 'r', encoding = 'utf-8')
# dictobject = open(lmlist, 'r', encoding = 'utf-8')
wordsobject = open(traintext, 'r')
dictobject = open(lmlist, 'r')

# get the dictionary data
dicts = Counter()
for line in dictobject.readlines():
    # print(line)
    dictArr = re.sub(r'\n', '', line).split()
    # print(dictArr)
    dicts.update(dictArr)
dd = dict(dicts)

ngram1d = len(dd)
print('1-grams in dict =', ngram1d)

# divide the ngrams
words = []
words2 = []
words3 = []
dict1 = Counter()
dict2 = Counter()
dict3 = Counter()

print('---counting ngrams---')

for line in wordsobject.readlines():
    wordArr = re.sub('\n', '', line).split()
    dict1.update(wordArr)
    words.extend(wordArr)
    for i in range(0, len(wordArr) - 1):
        word2 = [str(wordArr[i] + ' ' + wordArr[i + 1])]
        # print(word2)
        if i < len(wordArr) - 2:
            word3 = [str(wordArr[i] + ' ' + wordArr[i + 1] + ' ' + wordArr[i + 2])]
            # print(word3)
            dict3.update(word3)
        dict2.update(word2)

d1 = dict(dict1)  # uni-gram字典(训练语料得出的)
d2 = dict(dict2)  # bi-gram字典
d3 = dict(dict3)  # tri-gram字典
ngram1 = len(dict(dict1))
ngram2 = len(dict(dict2))  # bi-gram种数
ngram3 = len(dict(dict3))  # tri-gram种数


print('1-grams=', len(dict(dict1)))
print('2-grams=', len(dict(dict2)))
print('3-grams=', len(dict(dict3)))

# 定义trie树的数据结构


class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.is_word = False


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    '''
        The function of inserting a word into the trie.
        @param   string   word to be inserted
    '''

    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True

    '''
        The function of knowing if the word is in the trie.
        @param   string   the word to be found
        @returns   bool
    '''

    def search(self, word):
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中

    '''
        The function of knowing if if there is any word in the trie that starts with the given prefix.
        @param   string   the prefix to be found
        @returns   bool
    '''

    def starts_with(self, prefix):
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return False
        return True

    '''
        The function of getting the word that started with prefix.
        @param   string   the prefix to be found
        @returns   list    the words that started with prefix
    '''

    def get_start(self, prefix):

        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.extend(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list

        words = []
        if not self.starts_with(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)


# 将二元三元文法建立成trie树
print('---building tree---')
trie2 = Trie()
trie2b = Trie()
trie3 = Trie()
for key in d2:
    # print(key)
    trie2.insert(key)  # 二元正序树
    # print(key)
    keyb = str(key.split(' ')[1] + ' ' + key.split(' ')[0])
    # print(keyb)
    trie2b.insert(keyb)  # 二元逆序树

for key in d3:
    # print(key)
    trie3.insert(key)  # 三元树

# 计算公式中常数D（D = n1 / (n1 + 2*n2))
print('---calculating D---')
n11 = 0
n12 = 0
for key in d1:
    if d1[key] == 1:
        n11 += 1
    if d1[key] == 2:
        n12 += 1
db1 = n11 * 1.0 / (n11 + 2 * n12)


n21 = 0
n22 = 0
for key in d2:
    if d2[key] == 1:
        n21 += 1
    if d2[key] == 2:
        n22 += 1
db2 = n21 * 1.0 / (n21 + 2 * n22)

n31 = 0
n32 = 0
for key in d3:
    if d3[key] == 1:
        n31 += 1
    if d3[key] == 2:
        n32 += 1
db3 = n31 * 1.0 / (n31 + 2 * n32)

# 计算Pkn(wi)


def pwi(wi):
    xwi = 0
    wix = 0
    global db2
    global db1

    xwi = len(trie2b.get_start(str(wi + ' ')))  # *wi出现次数
    wix = len(trie2.get_start(str(wi + ' ')))  # wi*出现次数
    if wi == '<s>':
        p1 = 1
        arpap1 = -99
    elif (xwi == 0 and wi != '<s>'):
        p1 = db1 * 1.0 / (ngram1d * ngram2)
        arpap1 = log(p1, 10)  # 以10为底取对数
    else:
        p1 = xwi * 1.0 / ngram2
        arpap1 = log(p1, 10)  # 以10为底取对数
    # arpap1 = log(p1, 10) #以10为底取对数
    if wix == 0:
        # print('***',wi)
        bp1 = 1.0
    else:
        bp1 = (db2 * 1.0 / (d1[wi])) * wix
        # print(wi,bp1)
    arpabp1 = log(bp1, 10)
    return arpap1, arpabp1


# 计算Pkn(wi|wi-1)
def pwi_1wi(wi_1wi):
    global db2
    global db3
    wi_1wix = 0
    wi_1wi_a = wi_1wi.split(' ')[0]
    wi_1wi_b = wi_1wi.split(' ')[1]

    wi_1wix = len(trie3.get_start(str(wi_1wi + ' ')))  # wi-1wi*出现次数

    p2 = (max((d2[wi_1wi] - db2), 0.0)) / (d1[wi_1wi_a])
    bp2 = (db3 / (d2[wi_1wi])) * wi_1wix
    arpap2 = log(p2, 10)  # 以10为底取对数
    if bp2 == 0:
        # print('***',wi_1wi)
        return arpap2, ' '
    else:
        arpabp2 = log(bp2, 10)
        return arpap2, arpabp2

# 计算Pkn(wi|wi-2wi-1)


def pwi_2wi_1wi(wi_2wi_1wi):
    global db3
    wi_2wi_1wi_a = str(wi_2wi_1wi.split(' ')[0] + ' ' + wi_2wi_1wi.split(' ')[1])
    wi_2wi_1wi_b = str(wi_2wi_1wi.split(' ')[1] + ' ' + wi_2wi_1wi.split(' ')[2])

    p3 = max((d3[wi_2wi_1wi] - db3), 0.0) / (d2[wi_2wi_1wi_a])
    arpap3 = log(p3, 10)  # 以10为底取对数
    return arpap3

# 定义输出arpa格式的函数


def arpa(address):
    arpaname = address
    #arpaobject = open(arpaname, "w", encoding = 'utf-8')
    arpaobject = open(arpaname, "w")
    # arpaobject.write('\n')
    arpaobject.write('\\data\\')
    arpaobject.write('\n')
    arpaobject.write('ngram 1=' + str(ngram1d))
    arpaobject.write('\n')
    arpaobject.write('ngram 2=' + str(ngram2))
    arpaobject.write('\n')
    arpaobject.write('ngram 3=' + str(ngram3))
    arpaobject.write('\n')
    arpaobject.write('\n\\1-grams:')
    arpaobject.write('\n')

    print('---calculating probability---')

    # print('\n',end='',file=arpaobject)
    # print('\data\\',file=arpaobject)
    # print('ngram 1=',ngram1d,sep='',file=arpaobject)
    # print('ngram 2=',ngram2,sep='',file=arpaobject)
    # print('ngram 3=',ngram3,sep='',file=arpaobject)
    # print('\n\\1 - grams:',file=arpaobject)
    i = 0
    for key in dicts:
        i = i + 1
        if i % 10000 == 0:
            print('1-gram: ' + str(i))
        if key == '</s>':
            arpaobject.write(str(pwi(key)[0]) + '\t' + str(key))
            arpaobject.write('\n')
            #print(str(pwi(key)[0]) + str(key))

            # print(pwi(key)[0],key,file=arpaobject)
            # print(pwi(key)[0],key)
        else:
            arpaobject.write(str(pwi(key)[0]) + '\t' + str(key) + '\t' + str(pwi(key)[1]))
            arpaobject.write('\n')
            #print(str(pwi(key)[0]) + ' ' + str(key) + ' ' + str(pwi(key)[1]))
            # print(pwi(key)[0],key,pwi(key)[1],file=arpaobject)
            # print(pwi(key)[0],key,pwi(key)[1])
    print('1-gram finished!')
    # print('\n\\2 - grams:',file=arpaobject)
    arpaobject.write('\n\\2-grams:')
    arpaobject.write('\n')

    i = 0
    for key in d2:
        i = i + 1
        if i % 500000 == 0:
            print('2-gram: ' + str(i))
        if key[(len(key) - len('</s>')):len(key)] == '</s>':
            arpaobject.write(str(pwi_1wi(key)[0]) + '\t' + str(key))
            arpaobject.write('\n')
            #print(str(pwi_1wi(key)[0]) + ' ' + str(key))
            # print(pwi_1wi(key)[0],key,file=arpaobject)
            # print(pwi_1wi(key)[0],key)
        else:
            arpaobject.write(str(pwi_1wi(key)[0]) + '\t' + str(key) + '\t' + str(pwi_1wi(key)[1]))
            arpaobject.write('\n')
            #print(str(pwi_1wi(key)[0]) + ' ' + str(key) + ' ' + str(pwi_1wi(key)[1]))
            # print(pwi_1wi(key)[0],key,pwi_1wi(key)[1],file=arpaobject)
            # print(pwi_1wi(key)[0],key,pwi_1wi(key)[1])
    # print('\n\\3 - grams:',file=arpaobject)
    print('2-gram finished!')
    arpaobject.write('\n\\3-grams:')
    arpaobject.write('\n')

    i = 0
    for key in d3:
        i = i + 1
        if i % 1000000 == 0:
            print('3-gram: ' + str(i))
        arpaobject.write(str(pwi_2wi_1wi(key)) + '\t' + str(key))
        arpaobject.write('\n')
        #print(str(pwi_2wi_1wi(key)) + ' ' + str(key))
        # print(pwi_2wi_1wi(key),key,file=arpaobject)
        # print(pwi_2wi_1wi(key),key)
    print('3-gram finished!')
    arpaobject.write('\n')
    arpaobject.write('\\end\\')
    arpaobject.write('\n')
    arpaobject.close()


# 调用函数完成输出
arpa(sys.argv[3])

# 关闭文件
wordsobject.close()
dictobject.close()
