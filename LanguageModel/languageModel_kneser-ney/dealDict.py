#!/usr/bin/python
#coding = utf-8

'''
    This file achieves to deal the dictionary.
''' 
f_out = open('dict.txt', 'w', encoding='utf-8')
with open('dict_old.txt', 'r', encoding='utf-8') as f_in:
    lines = f_in.readlines()
    for line in lines:
        result = line.split(' /')[0]
        f_out.write(result)
        f_out.write('\n')
f_out.close()
