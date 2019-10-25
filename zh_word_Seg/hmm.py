# -*- coding:utf-8 -*-
# @Time    : 2019/10/23 14:55
# @Author  : Ray.X
import jieba
"""
jieba提供3种分词模式
    精确模式，试图将句子最精确地切开，适合文本分析；
    全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
    
"""
default = jieba.cut('在南京市长江大桥研究生命的起源', cut_all=False)  # 精确模式 也是默认模式
full = jieba.cut('在南京市长江大桥研究生命的起源', cut_all=True)  # 全模式
search = jieba.cut_for_search('在南京市长江大桥研究生命的起源')  # 搜索引擎模式
print('default', "/ ".join(default))
print('full', "/ ".join(full))
print('search', ", ".join(search))

"""
    词性标注
"""