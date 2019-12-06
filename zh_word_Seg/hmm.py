# -*- coding:utf-8 -*-
# @Time    : 2019/12/5 14:30
# @Author  : Ray.X
"""
    Use HMM for Chinese word segmentation
    训练语料为人民日报
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
        self.module_file = 'model/hmm_model.pkl'
        self.states_list = ['S', 'B', 'E', 'M']
        self.load_para = False  # 是否训练
        self.trans_dic = {}     # 状态转移概率
        self.emit_dic = {}      # 发射概率
        self.start_dic = {}     # 初始概率
        self.count_dic = {}     # 统计状态出现次数
        self.V = [{}]           # 统计最大概率路径

    def load_model(self, trained):
        """
        @param trained:  判断是使用模型，还是训练模型。
        @return: 训练模型则初始化清空初始概率、转移概率、发射概率
        """
        if trained:
            with open(self.module_file, 'rb') as f:
                self.start_dic = pickle.load(f)
                self.trans_dic = pickle.load(f)
                self.emit_dic = pickle.load(f)
                self.load_para = True
        else:  # 清空
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
            self.trans_dic[state] = {s: 0.0 for s in self.states_list}  # 初始化 状态转移矩阵
            # {'S': {'S': 0.0, 'B': 0.0, 'E': 0.0, 'M': 0.0},
            # 'B': {'S': 0.0, 'B': 0.0, 'E': 0.0, 'M': 0.0},
            # 'E': {'S': 0.0, 'B': 0.0, 'E': 0.0, 'M': 0.0},
            # 'M': {'S': 0.0, 'B': 0.0, 'E': 0.0, 'M': 0.0}}
            self.emit_dic[state] = {}                                   # 初始化 状态发射矩阵
            # {'S': {'，': 3.0, '新': 1.0, '的': 2.0},
            # 'B': {'中': 3.0, '儿': 1.0, '踏': 1.0},
            # 'E': {'年': 1.0, '亿': 1.0, '华': 1.0},
            # 'M': {'９': 1.0, '８': 1.0, '６': 1.0, '产': 1.0}}
            self.start_dic[state] = 0.0                                 # 初始化 初始概率
            # {'S': 0, 'B': 1, 'E': 0, 'M'}
            self.count_dic[state] = 0                                   # 初始化每个状态出现次数
            # {'S': 0, 'B': 1, 'E': 0, 'M'}

    def train(self, path):
        """
        训练HMM模型
        @param path: 训练语料地址
        """
        self.load_model(False)  # 重置概率矩阵

        def make_label(text):
            """
            生成标签
            @param text: 训练语料
            @return:
            """
            out_text = []
            if len(text) == 1:
                out_text.append('S')  # 单字标记为S
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']  # 多字标记为  BE BME BMME...
            return out_text

        self.init_parameters()
        line_num = -1  # 从0行开始

        words = set()  # 构建无序无重的字、标点集合
        with open(path, encoding='UTF-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()  # 清除头尾字符 默认空格、换行
                if not line:
                    continue

                word_list = [i for i in line if i != ' ']
                words |= set(word_list)  # 合并字符集

                line_list = line.split()  # 按空格分割语料 语料为标注完成的 词语
                line_state = []
                for w in line_list:
                    line_state.extend(make_label(w))  # 生成每个字的标签集

                assert len(word_list) == len(line_state)  # 断言 条件为True正常执行
                for i, v in enumerate(line_state):  # 遍历Data 生成[(index, state)]
                    self.count_dic[v] += 1      # 统计每个状态的次数
                    if i == 0:
                        self.start_dic[v] += 1  # 统计每一个句子第一个字的状态，用于计算初始状态概率
                    else:
                        self.trans_dic[line_state[i - 1]][v] += 1  # 统计状态间的转移次数，用于计算转移概率 状态——>状态
                        self.emit_dic[line_state[i]][word_list[i]] = \
                            self.emit_dic[line_state[i]].get(word_list[i], 0) + 1.0  # 统计每个状态下出现的字次数，用于计算发射概率 状态->字
        # 计算初始概率  句子的第一个字属于 k 状态出现的次数/总的句子数
        self.start_dic = {k: v * 1.0 / line_num for k, v in self.start_dic.items()}
        # 计算转移概率  k状态的前一个状态k1出现的次数v1/k状态出现的次数
        self.trans_dic = {k: {k1: v1 / self.count_dic[k] for k1, v1 in v.items()} for k, v in self.trans_dic.items()}
        # 计算发射概率 加1平滑防止某个字出现的次数为0 则分子为0
        # k状态下 k1字出现的次数/k状态出现的次数
        self.emit_dic = {k: {k1: (v1 + 1) / self.count_dic[k] for k1, v1 in v.items()} for k, v in self.emit_dic.items()}
        # 储存模型为dump
        with open(self.module_file, 'wb') as f:
            pickle.dump(self.start_dic, f)
            pickle.dump(self.trans_dic, f)
            pickle.dump(self.emit_dic, f)

        return self

    def veterbi(self, obs, states, start_p, trans_p, emit_p):
        """
        使用veterbi算法计算最大概率路径
        @param obs:     输入
        @param states:  状态集
        @param start_p: 初始概率
        @param trans_p: 转移概率矩阵
        @param emit_p:  发射概率矩阵
        @return:
        """
        self.V = [{}]    # V[t][状态] = 路径概率
        path = {}   # 中间变量，表路径上的当前状态

        # 初始化初始状态
        for k in states:
            self.V[0][k] = start_p[k] * emit_p[k].get(obs[0], 0)  # k状态初始概率 * k状态下 句子第一个字 出现的概率
            path[k] = [k]  # {'S':[...]}

        # 对t>1的节点计算最大概率路径
        for t in range(1, len(obs)):
            self.V.append({})
            newpath = {}

            # 判断该字是否在发射矩阵中
            neverSeen = obs[t] not in emit_p['S'].keys() and obs[t] not in emit_p['M'].keys() and obs[t] not in \
                emit_p['E'].keys() and obs[t] not in emit_p['B'].keys()

            for k in states:
                emitP = emit_p[k].get(obs[t], 0) if not neverSeen else 1.0  # 未知字单独成词，不存的概率在设为 1.0
                # (最大概率，概率最大的前一个状态) = max（前一个状态k0的概率 * k0到k的转移概率 * 这个字 出现在k状态下 概率）
                (prob, state) = max([(self.V[t-1][k0] * trans_p[k0].get(k, 0) * emitP, k0) for k0 in states
                                     if self.V[t-1][k0] > 0])

                self.V[t][k] = prob  # 记录最大概率
                newpath[k] = path[state] + [k]  # 记录路径

            path = newpath  # 初始化，不保留旧路径

        # 得到最大概率路径，计算最后第二个字的状态最大概率
        if emit_p['M'].get(obs[-1], 0) > emit_p['S'].get(obs[-1], 0):
            # 如果最后一个字是词尾的概率大于单词，最后第二个字的状态只可能是词中或者词尾（M > S 不能说一定是M）
            (prob, state) = max([(self.V[len(obs) - 1][k], k) for k in ('E', 'M')])
        else:
            # 否则都有可能
            (prob, state) = max([(self.V[len(obs) - 1][k], k) for k in states])
        print(path, state)
        print_dptable(self.V)
        return prob, path[state]

    def cut(self, text):
        """
        分词接口，调用veterbi
        @param text: 输入
        @return:
        """
        if not self.load_para:
            self.load_model(os.path.exists(self.module_file))
        prob, max_path = self.veterbi(text, self.states_list, self.start_dic, self.trans_dic, self.emit_dic)
        print(prob, max_path)
        start, then = 0, 0
        for i, char in enumerate(text):
            pos = max_path[i]
            if pos == 'B':
                start = i
            elif pos == 'E':
                yield text[start: i+1]
                print('1', text[start: i+1])
            elif pos == 'M':
                then = i+1
            elif pos == 'S':
                yield char
                print('2', char)
                then = i+1
        if then < len(text):
            yield text[then:]
            print('3', text[then:])

        # yield 即return的同时


# 打印路径概率表
def print_dptable(V):
    print("    ", end=' ')
    for i in range(len(V)):
        print("%7d" % i, end=' ')
    print()

    for y in list(V[0].keys()):
        print("%.5s: " % y, end=' ')
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=' ')
        print()


hmm = HMM()
# hmm.train('../Data/CWS/trainCorpus.txt_utf8')
res = hmm.cut('南京市长江大桥')
print(str(list(res)))
# [{'S': 0.00024068043618878063, 'B': 0.0008479575355477435, 'E': 0.0, 'M': 0.0},
#  {'S': 2.597703368715881e-08, 'B': 5.829385508478131e-09, 'E': 1.8054004390160279e-06, 'M': 2.994752366629435e-07},
#  {'S': 9.838524382946052e-10, 'B': 1.6974514448106187e-09, 'E': 5.794565751422634e-10, 'M': 3.5958846294198544e-11},
#  {'S': 3.886964550765146e-13, 'B': 4.18999169959862e-13, 'E': 7.021903255143946e-12, 'M': 2.4633073463114406e-13},
#  {'S': 1.1929121178105033e-15, 'B': 1.2449982203103038e-15, 'E': 1.6152119319628026e-16, 'M': 8.934275095138284e-17},
#  {'S': 1.9171797519360457e-18, 'B': 2.9217898225125463e-18, 'E': 4.6433599609509474e-18, 'M': 1.3333437077827637e-18},
#  {'S': 2.4660655016412213e-22, 'B': 9.879329881241226e-23, 'E': 2.1188370123793454e-22, 'M': 9.902992826341432e-24}]
# {'S': ['B', 'E', 'B', 'E', 'B', 'E', 'S'],
#  'B': ['B', 'E', 'B', 'E', 'B', 'E', 'B'],
#  'E': ['B', 'E', 'B', 'E', 'S', 'B', 'E'],
#  'M': ['B', 'E', 'B', 'E', 'B', 'M', 'M']}
