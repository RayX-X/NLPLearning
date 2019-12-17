# -*- coding:utf-8 -*-
# @Time    : 2019/11/21 19:45
# @Author  : Ray.X
import zipfile
import lxml.etree
import re
"""
    获取xml中的有效文本 content 数据 keywords 标签
"""
with zipfile.ZipFile(r'D:\C\NLP\Data\ted_zh-cn-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_zh-cn-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))  # 获取<content>标签下的文字
z.close()
del doc, z

input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
input_text_noparens = re.sub(r'（[^）]*）', '', input_text_noparens)

sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in re.split('[。？！]', m.groupdict()['postcolon']) if sent)
del input_text_noparens, input_text

sentences_strings_ted = [re.sub(r'[^\w\s]', '', sent) for sent in sentences_strings_ted]
sentences_strings_ted = [re.sub(r'[a-zA-Z0-9]', '', sent) for sent in sentences_strings_ted]
sentences_strings_ted = filter(None, sentences_strings_ted)
data = ' '.join([re.sub(r'\s', '', sent) for sent in sentences_strings_ted]).split(' ')
del sentences_strings_ted

datax = [' '.join(sent).split(' ') for sent in data]
# del data
from nltk import ngrams, FreqDist
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated, MLE

# penta_grams =[list(ngrams(pad_both_ends(sent, 2), 2)) for sent in datax]
# vocaby = list(flatten(pad_both_ends(sent, n=2) for sent in datax))

train, vocab = padded_everygram_pipeline(5, datax)
lm = KneserNeyInterpolated(5)
lm.fit(train, vocab)

# In[]
# freq_dist = FreqDist(penta_grams)
# test = '我想带你们体验一下，我们所要实现的“信任”的感觉。'
# x = lm.perplexity(ngrams(list(pad_both_ends(test, 5)), 5))
#
# # In[]
# sent = lm.generate(4, random_seed=3)
