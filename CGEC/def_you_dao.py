# -*- coding:utf-8 -*-
# @Time    : 2019/11/21 15:52
# @Author  : Ray.X
import pandas as pd
import csv

# 读原始数据
datalist = open(r"../Data/CGEC/data.train", encoding='UTF-8').readlines()
dataline = [datalist[i].replace('\n', '').split('\t') for i in range(len(datalist))]
del datalist
# 处理成(srcSent, tgtSent)对集合
newdata = []
for i in range(len(dataline)):
    if dataline[i][1] == '0':
        newdata.append([dataline[i][2], dataline[i][2]])
    else:
        for j in range(3, len(dataline[i])):
            newdata.append([dataline[i][2], dataline[i][j]])
del dataline
# In[] (srcSent, tgtSent)过滤

