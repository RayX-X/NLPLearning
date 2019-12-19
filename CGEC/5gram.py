# -*- coding:utf-8 -*-
# @Time    : 2019/11/21 19:45
# @Author  : Ray.X
#
# from nltk.lm import KneserNeyInterpolated
# from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
# from nltk import ngrams
# import zipfile
# import lxml.etree
# import re
import joblib
# """
#     获取xml中的有效文本 content 数据 keywords 标签
# """
# with zipfile.ZipFile(r'D:\C\NLP\Data\ted_zh-cn-20160408.zip', 'r') as z:
#     doc = lxml.etree.parse(z.open('ted_zh-cn-20160408.xml', 'r'))
# input_text = '\n'.join(doc.xpath('//content/text()'))  # 获取<content>标签下的文字
# z.close()
#
# del doc, z
#
# input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
# input_text_noparens = re.sub(r'（[^）]*）', '', input_text_noparens)
#
# sentences_strings_ted = []
# for line in input_text_noparens.split('\n'):
#     m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
#     sentences_strings_ted.extend(sent for sent in re.split('[。？！]', m.groupdict()['postcolon']) if sent)
#
# del input_text_noparens, input_text
#
# sentences_strings_ted = [re.sub(r'[^\w\s]', '', sent) for sent in sentences_strings_ted]
# sentences_strings_ted = [re.sub(r'[a-zA-Z0-9]', '', sent) for sent in sentences_strings_ted]
# sentences_strings_ted = filter(None, sentences_strings_ted)
# data = ' '.join([re.sub(r'\s', '', sent) for sent in sentences_strings_ted]).split(' ')
# datax = [' '.join(sent).split(' ') for sent in data]
#
# del sentences_strings_ted, data

# 训练 5-gram
# lm = KneserNeyInterpolated(3)
# train, vocab = padded_everygram_pipeline(3, datax)
# lm.fit(train, vocab)

# del train, vocab, datax
# 困惑度测试
# test = '我想带你们体验一下，我们所要实现的“信任”的感觉。'
# sent_list = re.sub(r'[^\w\s]', '', test)
# sent_list = ','.join(sent_list).split(',')
# text = list(ngrams(pad_both_ends(sent_list, 5), 5))
#
# entropy = lm.entropy(text)  # 交叉熵
# perplexity = lm.perplexity(text)  # 困惑度
# print('交叉熵:%f' % entropy, '困惑度:%f' % perplexity)
# 储存模型  ... 以下内容 内存不足跑不起来 去 Colaboratory 或者 kaggle 跑蹭谷歌服务器
# joblib.dump(lm, '3gram.pkl')
# In[]
# # 测试储存的模型
# kn = joblib.load('kn_5gram.pkl')
#
# kn_entropy = kn.entropy(text)  # 交叉熵
# kn_perplexity = kn.perplexity(text)  # 困惑度
# print('KN交叉熵:%f' % kn_entropy, 'KN困惑度:%f' % kn_perplexity)
lm = joblib.load('3gram.pkl')
# In[]

print(lm.perplexity([('我', '想', '你'), ('想', '你', '啦')]))