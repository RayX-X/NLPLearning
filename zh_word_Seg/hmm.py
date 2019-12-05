# -*- coding:utf-8 -*-
# @Time    : 2019/12/5 14:30
# @Author  : Ray.X
"""
    Use HMM for Chinese word segmentation
"""
import os
import pickle


class HMM:
    def __init__(self):
        """
            定义模型保存位置
            状态集合[S, B, E, M] S 单独成词 B词首E词中M词尾
            加载参数，判断是否需要重新训练
        """
        self.module_file = 'hmm_model.pkl'
        self.states_list = ['S', 'B', 'E', 'M']
        self.load_para = False  # 是否训练
        self.trans_dic = {}     # 状态转移概率
        self.emit_dic = {}      # 发射概率
        self.start_dic = {}     # 初始概率
        self.text = ''          # 训练语料

    def load_model(self, trained):
        """
        @param trained:  判断是使用模型，还是训练模型。
        @return: 训练模型则初始化清空初始概率、转移概率、发射概率
        """
        if trained:
            with open(self.module_file, 'rb') as f:
                self.trans_dic = pickle.load(f)
                self.emit_dic = pickle.load(f)
                self.start_dic = pickle.load(f)
                self.load_para = True
        else:
            self.load_para = False
            self.trans_dic = {}
            self.emit_dic = {}
            self.start_dic = {}

    def init_parameters(self):
        """
        初始化参数
        @return:
        """
        for state in self.states_list:
            self.trans_dic[state] = {s: 0.0 for s in self.states_list}
            self.emit_dic[state] = {}
            self.start_dic[state] = 0.0

    def make_label(self):
        """
        生成标签
        @param text: 训练语料
        @return:
        """

        out_text = []
        if len(self.text) == 1:
            out_text.append('S')  # 单字标记为S
        else:
            out_text += ['B'] + ['M'] * (len(self.text) - 2) + ['E']  # 多字标记为  BE BME BMME...
        return out_text

    def train(self, path):
        """
        训练HMM模型
        @param path: 训练语料地址
        """
        self.load_model(False)  # 重置概率矩阵
        count_dic = {}  # 统计状态出现次数

        self.init_parameters()
        line_num = -1  # 从0行开始

        words = set()
        with open(path, encoding='UTF-8') as f:
            for line in f:
                line_num += 1
                line = line.split()
                if not line:
                    continue
                word_list = [i for i in line if i != '']
                words |= set(word_list)
