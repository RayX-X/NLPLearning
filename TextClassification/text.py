# -*- coding:utf-8 -*-
# @Time    : 2019/10/14 16:20
# @Author  : Ray.X
import re
from nltk.corpus import stopwords
from nltk.tokenize.stanford import StanfordTokenizer


str = '我想和你们分享一个故事， 关于我差点被绑到一辆 红色马自达后备箱的故事。 那是从设计学校毕业之后第二天， 我在后院里弄了个旧货拍卖。 这个家伙开着红色马自达过来了， 他停了车并开始打量我的东西。 最后，他买了一件我的艺术作品。 我得知他今晚在这个镇上 是孤身一人， 他正在进行加入 美国和平队之前的 穿越美国的汽车旅行。 于是我请他出去喝了一杯， 他跟我聊到关于 他想要改变世界的所有宏图大略。'



# input_text_noparens = re.sub(r'\([^)]*\)|（[^）]*\)|（[^）]*）', '', str)
# print(input_text_noparens)

# a = re.sub(r'[^\w \']', '', "what's rong?>\n<';~-_")
# print(a)

stopwords = stopwords.words('english')