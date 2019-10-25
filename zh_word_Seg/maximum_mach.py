# -*- coding:utf-8 -*-
# @Time    : 2019/10/21 11:12
# @Author  : Ray.X
# 初始化
import lxml.etree
import re
# df = open('../Data/dict/baidu_dict.txt', 'r', encoding='utf-8').readlines()
# word_dict = []
# for line in df:
#     word_dict.append(line.split()[1])
word_dict = ['研究', '研究生', '生命', '起源', '南京市', '市长', '长江', '大桥', '长江大桥']
test_str = '在南京市长江大桥研究生命的起源'
# 遍历分词词典，获得最大分词长度
MaxLen = 0
for key in word_dict:
    if len(key) > MaxLen:
        MaxLen = len(key)


def forward_mm():
    """
    正向最大匹配 MM
    :return:
    """
    foward_out = []
    n = 0
    while n < len(test_str):
        matched = 0
        # range(start, stop, step)，根据start与stop指定的范围以及step设定的步长 step=-1表示去掉最后一位
        for i in range(MaxLen, 0, -1):  # i等于max_chars到1 3-2-1
            w = test_str[n: n + i]  # 截取文本字符串n到n+1位
            # 判断所截取字符串是否在分词词典内
            if w in word_dict:
                foward_out.append(w)
                matched = 1
                n = n + i
                break
        if not matched:  # 等于 if matched == 0
            foward_out.append(test_str[n])
            n = n + 1
    print('正向最大匹配 FMM:\n', foward_out)
    return foward_out


def reverse_mm():
    """
    逆向最大匹配 RMM
    :return:
    """
    reverse_out = []
    n = len(test_str)
    while 0 < n:
        matched = 0
        # range([start,] stop[, step])，根据start与stop指定的范围以及step设定的步长 step=-1表示去掉最后一位
        for i in range(MaxLen, 0, -1):  # i等于max_chars到1
            w = test_str[n - i: n]  # 截取文本字符串n-i到n位
            # 判断所截取字符串是否在分词词典内
            if w in word_dict:
                reverse_out.append(w)
                matched = 1
                n = n - i
                break
        if matched == 0:
            reverse_out.append(test_str[n - 1])
            n = n - 1
    print('逆向最大匹配 RMM:\n', list(reversed(reverse_out)))  # 因为是逆向所以最终结果需要反向
    return list(reversed(reverse_out))


def bi_mm():
    """
    双向最大匹配 BMM
    :return:
    """
    fmm = forward_mm()
    rmm = reverse_mm()
    # 单字词个数
    f_single_word = 0
    r_single_word = 0
    # 总词数
    tot_fmm = len(fmm)
    tot_rmm = len(rmm)
    # 未登录词
    oov_fmm = 0
    oov_rmm = 0
    # 罚分，罚分值越低越好
    score_fmm = 0
    score_rmm = 0
    # 如果正向和反向结果一样，返回任意一个
    if fmm == rmm:
        bmm = rmm
    else:  # 分词结果不同，返回单字数、非字典词、总词数少的那一个
        for w in fmm:
            if len(w) == 1:
                f_single_word += 1
            if w not in word_dict:
                oov_fmm += 1
        for w in rmm:
            if len(w) == 1:
                r_single_word += 1
            if w not in word_dict:
                oov_rmm += 1

        # 可以根据实际情况调整惩罚分值
        # 这里都罚分都为1分
        # 非字典词越少越好
        if oov_fmm > oov_rmm:
            score_fmm += 1
        elif oov_fmm < oov_rmm:
            score_rmm += 1
        # 总词数越少越好
        if tot_fmm > tot_rmm:
            score_fmm += 1
        elif tot_fmm < tot_rmm:
            score_rmm += 1
        # 单字词越少越好
        if f_single_word > r_single_word:
            score_fmm += 1
        elif f_single_word < r_single_word:
            score_rmm += 1

        # 返回罚分少的那个
        if score_fmm < score_rmm:
            bmm = fmm
        else:
            bmm = rmm
    print('双向最大匹配 BMM:\n', bmm)
    print('oov_fmm:%d' % oov_fmm, 'oov_rmm:%d' % oov_rmm)
    print('tot_fmm:%d' % tot_fmm, 'tot_rmm:%d' % tot_rmm)
    print('f_single_word:%d' % f_single_word, 'r_single_word:%d' % r_single_word)


if __name__ == "__main__":
    bi_mm()
