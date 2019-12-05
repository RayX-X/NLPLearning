# -*- coding:utf-8 -*-
# @Time    : 2019/10/22 9:12
# @Author  : Ray.X

line_num = -1
words = set()
with open('../Data/news/C000008/10.txt', encoding='GBK') as f:
    for line in f:
        print(line)
        line_num += 1
        line = line.split()
        if not line:
            print(line)
            continue
        word_list = [i for i in line if i != '']
        print(word_list)
        words |= set(word_list)
        print(words, '\n..........')
