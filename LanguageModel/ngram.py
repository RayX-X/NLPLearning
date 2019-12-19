# -*- coding:utf-8 -*-
# @Time    : 2019/12/9 18:52
# @Author  : Ray.X
"""
    Realization unigram bigram trigram and more 字粒度
"""
import re
from collections import defaultdict
from math import log
import zipfile
import lxml.etree
from nltk.probability import ConditionalFreqDist, FreqDist
import joblib


def pre_data():
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
    fin_data = [' '.join(sent).split(' ') for sent in data]
    del sentences_strings_ted, data

    return fin_data


class NGram:
    def __init__(self, n):
        """
        定义N元模型参数
        nltk的 ConditionalFreqDist 很好用就没有复写
        参考 from nltk.probability import ConditionalFreqDist, FreqDist
        @param n: 定义 ngram 元
        """
        self.N = n
        self.counter = defaultdict(ConditionalFreqDist)
        self.counter[1] = self.unigrams = FreqDist()

    def prepare(self, sents):
        """
        准备数据 分句在分字，句子头尾增加<BOS><EOS>
        @return:
        """
        n = self.N
        left = ['<BOS>']
        right = ['<EOS>']
        sents = list(left * (n - 1) + sent + right * (n - 1)for sent in sents)
        return sents

    def fit(self, sents):
        """
        训练函数 其实就是统计所有 1-n 元 ngram
        self.counter[n][gram[:-1]][gram[-1]]
        n =[1:n] 代表几元组
        [gram[:-1]] 代表当前元组的(wi-n-1,...wi-1)
        [gram[-1]]代表(wi-n-1,...wi-1)后存在的(w`)
        例：self.counter=
                    3:[(a, b):[c, d, e], (a, c):[c, d, e]]
                    2:[(a):[b,c,d]]
                    1:a,b,c,d
        @param sents: 输入形式[[1,2,3,4,5],[6,7,8,9],[10,11]] 字粒度
        @return:
        """
        ready = self.prepare(sents)
        n = 1
        while n <= self.N:
            for sent in ready:
                for i in range(len(sent) - n + 1):
                    gram = tuple(sent[i:i + n])
                    if n == 1:
                        self.unigrams[gram[0]] += 1
                        continue
                    self.counter[n][gram[:-1]][gram[-1]] += 1
            n += 1

    def alpha_gamma(self, word, context, d=0.1):
        """
        计算 kneser_ney平滑公式的两个部分
        @return:
        """
        prefix_counts = self.counter[len(context) + 1][context]

        alpha = max(prefix_counts[word] - d, 0.0) / prefix_counts.N()
        s = sum(1.0 for c in prefix_counts.values() if c > 0)
        gamma = d * s / prefix_counts.N()
        # print(s)
        # print(alpha, gamma)

        return alpha, gamma

    def kneser_ney(self, word, context):
        """
        return self.kneser_ney(word, context[1:]) 的目的是迭代
        pkn(w[i-n+1],...wi) -> pkn(w[i-n+2],...wi)...pkn(unigrm) ->
        @return:
        """
        if not context:
            return 1.0 / len(self.unigrams)
        alpha, gamma = self.alpha_gamma(word, context)
        return alpha + gamma * self.kneser_ney(word, context[1:])

    def perplexity(self, test_ngrams):
        """
        困惑度
        输入为 ngram 形式
        @return:
        """
        logs = sum(log(self.kneser_ney(ngram[-1], ngram[:-1]), 2) for ngram in test_ngrams)
        entropy = -1 * logs / len(test_ngrams)
        perplexit = pow(2.0, entropy)
        return perplexit


if __name__ == "__main__":
    train_data = pre_data()
    lm = NGram(3)
    lm.fit(train_data)
    # joblib 储存、复用模型
    # joblib.dump(lm, 'ngram.pkl')
    # lm = joblib.load('ngram.pkl')
    perplexity = lm.perplexity([('我', '想', '你'), ('想', '你', '啦')])
    print(perplexity)
