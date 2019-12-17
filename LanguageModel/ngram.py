# -*- coding:utf-8 -*-
# @Time    : 2019/12/9 18:52
# @Author  : Ray.X
"""
    Realization unigram bigram trigram and more 字粒度
"""
import re
from itertools import chain
from nltk.lm import KneserNeyInterpolated
from collections import Counter



class Ngram:
    def __init__(self, n):
        """
        定义N元模型参数
        """
        self.N = n
        self.count_up = Counter()
        self.count_down = Counter()
        self.unigram = Counter()

    def prepare(self, sents):
        """
        准备数据 分句在分字，句子头尾增加<s></s>
        @return:
        """
        left = ['<BOS>']
        right = ['<EOS>']
        sents = (left*(self.N - 1) + list(sents) + right*(self.N - 1))

        return iter(sents)

    def n_gram(self, sents):
        """
        N元模型
        @return:
        """
        history = []
        n = self.N
        while n > 1:
            try:
                next_sent = next(sents)
            except StopAsyncIteration:
                return
            history.append(next_sent)
            n -= 1
        for sent in sents:
            history.append(sent)
            yield tuple(history)
            del history[0]

    def count(self, ngrams):
        """
        count_up: C(wi-n-1,...,wi-1,wi)
        count_down: C(wi-n-1,...,wi-1)
        @return:
        """

        for i in range(len(ngrams)):
            for ngram in ngrams[i]:
                if len(ngram) == 1:
                    self.unigram[ngram] += 1
                    continue
                self.count_up[ngram] += 1
                self.count_down[ngram[:-1]] += 1

        # self.count_up = self.count_up.most_common()
        # self.count_down = self.count_down.most_common()

    def ckn(self):
        """
        选择合适的 low ngram
        @return:
        """
        return 1

    def kneserNey(self, d=0.75):
        """
        Kneser Ney 平滑
        @return:
        """
        pkn = []
        for i in range(len(self.count_up)):
            if i == 0:
                pkn[i] = max(self.count_up[i] - d, 0) / self.ckn()
            else:
                pkn[i] = max(self.count_up[i] - d, 0) / self.ckn() + d/self.ckn()*pkn[i-1]

    def fit(self, sents):
        """
        训练函数
        @param sents: 输入形式[[1,2,3,4,5],[6,7,8,9],[10,11]]
        @return:
        """
        ngram_data = [list(self.n_gram(self.prepare(sent))) for sent in sents]
        Vocabulary = [chain.from_iterable(self.prepare(sent) for sent in sents)]
        self.count(ngram_data)

        # lm = KneserNeyInterpolated(2)
        # lm.fit(ngram_data, Vocabulary)
        # print(lm.perplexity([('当', '你'), ('遇', '到')]))


if __name__ == "__main__":
    sentences = '我想和你们分享一个故事，关于我差点被绑到一辆红色马自达后备箱的故事。' \
                '那是从设计学校毕业之后第二天，我在后院里弄了个旧货拍卖。' \
                '这个家伙开着红色马自达过来了，他停了车并开始打量我的东西。' \
                '最后，他买了一件我的艺术作品。\n' \
                '我得知他今晚在这个镇上是孤身一人，他正在进行加入美国和平队之前的穿越美国的汽车旅行。' \
                '于是我请他出去喝了一杯，他跟我聊到关于他想要改变世界的所有宏图大略。'
    sent_list = sentences.split('。')
    sent_list = [re.sub(r'[^\w\s]', '', sent.strip()) for sent in sent_list]
    sent_list = [' '.join(w).split(' ') for w in sent_list]
    while [""] in sent_list:
        sent_list.remove([""])

    lm = Ngram(3)
    lm.fit(sent_list)



