# -*- coding:utf-8 -*-
# @Time    : 2019/10/25 19:16
# @Author  : Ray.X
import re

# In[] re.match()
"""
re.match(pattern, string, flags=0) 从左至右匹配
    pattern: 匹配的正则表达式
    string: 要匹配的字符串
    flags: 标志位，用于控制匹配方式
成功返回匹配 MatchObject 对象，否则返回None
MatchObject 属性
    string      匹配时使用的文本
    re          匹配时使用的Pattern对象。
    pos         文本中正则表达式开始搜索的索引
    lastindex   最后一个被捕获的分组在文本中的索引。如果没有被捕获的分组，将为None
    lastgroup:  最后一个被捕获的分组的别名。如果这个分组没有别名或者没有被捕获的分组，将为None。
MatchObject 方法
    group() 返回被 RE 匹配的字符串 获得一个或多个分组截获的字符串；指定多个参数时将以元组形式返回
    groups([default]) 以元组形式返回全部分组截获的字符串 相当于调用group(1,2,…)default表示没有截获字符串的组以这个值替代，默认为None。
    groupdict([default])返回以有别名的组的别名为键、以该组截获的子串为值的字典，没有别名的组不包含在内。default含义同上
    start(group) 返回指定的组截获的子串在string中的起始索引（子串第一个字符的索引）。group默认值为0。
    end(group)   返回指定的组截获的子串在string中的结束索引（子串最后一个字符的索引+1）。group默认值为0。
    span(group)  返回一个元组包含匹配 (开始,结束) 的位置 (start(group), end(group))。
"""
match = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello world!')
print(match.string)  # hello world!
print(match.re)  # re.compile('(\\w+) (\\w+)(?P<sign>.*)')
print(match.pos)  # 0
print(match.lastindex)  # 3
print(match.lastgroup)  # sign
print(match.group(0, 1))  # ('hello world!', 'hello')
print(match.groups())  # ('hello', 'world', '!')
print(match.groupdict())  # {'sign': '!'}
print(match.start(1))  # 0
print(match.end(1))  # 5
print(match.span(1))  # (0, 5)

# In[] re.split()
"""
re.split(pattern, string, maxsplit=0, flags=0) 按照能够匹配的子串将字符串分割后返回列表
    pattern: 匹配的正则表达式
    string: 要匹配的字符串
    flags: 标志位，用于控制匹配方式
    maxsplit 分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次数。
成功返回匹配 objec 对象，否则返回None
"""

split = re.split(r'\W', '南京,长江,大桥。')
print(split)  # ['南京', '长江', '大桥', '']

split = re.split(r'(\W+)', '南京,长江,大桥。')
print(split)  # ['南京', ',', '长江', ',', '大桥', '。', '']

split = re.split(r'(\W+)', '南京,长江,大桥。', 1)
print(split)  # ['南京', ',', '长江,大桥。']

# In[]
"""
Pattern对象是一个编译好的正则表达式，通过Pattern提供的一系列方法可以对文本进行匹配查找。
Pattern不能直接实例化，必须使用re.compile()进行构造。

pattern: 匹配的正则表达式
flags:  编译时用的匹配模式。数字形式。
groups: 表达式中分组的数量。
groupindex: 以表达式中有别名的组的别名为键、以该组对应的编号为值的字典，没有别名的组不包含在内。
"""
p = re.compile(r'(\w+) (\w+)(?P<sign>.*)', re.DOTALL)
print("p.pattern:", p.pattern)  # p.pattern: (\w+) (\w+)(?P<sign>.*)
print("p.flags:", p.flags)  # 48
print("p.groups:", p.groups)  # 3
print("p.groupindex:", p.groupindex)  # {'sign': 3}

# In[]
"""
re.search(pattern, string, flags)
这个方法用于查找字符串中可以匹配成功的子串。从string的pos下标处起尝试匹配pattern，
如果pattern结束时仍可匹配，则返回一个Match对象；若无法匹配，则将pos加1后重新尝试匹配；直到pos=endpos时仍无法匹配则返回None。
pos和endpos的默认值分别为0和len(string))；re.search()无法指定这两个参数，参数flags用于编译pattern时指定匹配模式。
"""
search = re.search(r'world', 'hello world!')
print(search.group())  # world
# In[]
"""
re.findall(pattern, string, flags)
搜索string，以列表形式返回全部能匹配的子串。
"""
print(re.findall(r'\([^)]*\)', '(南)(京)(长江)大桥'))  # ['(南)', '(京)', '(长江)']
# In[]
"""
re.finditer(pattern, string, flags):
搜索string，返回一个顺序访问每一个匹配结果（Match对象）的迭代器。
"""
[print(w.group()) for w in re.finditer(r'\([^)]*\)', '(南)(京)(长江)大桥')]  # (南) (京) (长江)
# In[]
"""
re.sub(pattern, repl, string, count): 
使用repl替换string中每一个匹配的子串后返回替换后的字符串。
当repl是一个字符串时，可以使用\id或\g<id>、\g<name>引用分组，但不能使用编号0。
当repl是一个方法时，这个方法应当只接受一个参数（Match对象），并返回一个字符串用于替换（返回的字符串中不能再引用分组）。
count用于指定最多替换次数，不指定时全部替换。
re.sub(pattern, repl, string, count): 
返回 (sub(repl, string, count), 替换次数)。
"""
print(re.sub(r'\([^)]*\)', ',', '(南)(京)(长江)大桥', 2))  # ,,(长江)大桥
# 当repl是一个方法时
# 将匹配的数字乘以 2


def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)


s = 'A23G4HFD567'
print(re.sub(r'(?P<value>\d+)', double, s))

print(re.subn(r'\([^)]*\)', ',', '(南)(京)(长江)大桥', 2))  # (',,(长江)大桥', 2)

# In[]
large = re.match(r'HELLO', 'hello world!', re.I)
low = re.match(r'HELLO', 'hello world!')
print(large.group())  # hello
print(low.group())  # AttributeError: 'NoneType' object has no attribute 'group'
# In[]
s = '1102231990xxxxxxxx'
res = re.search(r'(?P<province>\d{3})(?P<city>\d{3})(?P<born_year>\d{4})', s)
print(res.groupdict())  # {'province': '110', 'city': '223', 'born_year': '1990'}
