# -*- coding:utf-8 -*-
# @Time    : 2019/10/7 10:36
# @Author  : Ray.X

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, FastText
from gensim import corpora
import lxml.etree
import zipfile
import urllib.request
import numpy as np
import os
from random import shuffle
import re
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show
from sklearn.cluster import KMeans

print('Part1 : 文本预处理')

with zipfile.ZipFile(r'D:\C\NLP\Data\ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

input_text = '\n'.join(doc.xpath('//content/text()'))  # 获取<content>标签下的文字
i = input_text.find("Hyowon Gweon: See this?")
j = input_text.find("yowon Gweon: See this?")
print(i, '->', j, '->', input_text[i - 20:i + 150])

input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)  # 使用正则表达式删除所有带圆括号的字符串：
k = input_text_noparens.find("Hyowon Gweon: See this?")
print(k, '->', input_text_noparens[i - 20:i + 150])
"""
    现在，让我们尝试删除出现在一行开头的发言者的姓名，
    通过删除“`<up to 20 characters>：`”格式的片段
    ^(?:(?P<precolon>[^:]{,20}):) 意思为 匹配 从开头到'：'之间 除'：'外字符数小于等于20的字符串，并将所匹配的组命名为 preconlon
        (字符包括空格)
        (?P<name>exp) 为命名一个分组
        (?:(exp):)匹配不获取分组
        [^:] 匹配除'：'外的字符
        {,20} 最多匹配前20个字符 超过则不匹配
        ^ 指非或者开头
        例: "And here's the thing: it's not really a joke."  匹配到"And here's the thing"
            "And here is the thing: it's not really a joke." 匹配到 None
            "Here's the thing: it's not really a joke."  匹配到 "Here's the thing"
    ? 即之后的内容可有可无i
    (?P<postcolon>.*)$ 指'：'后的所有内容 并命名为postcolon组
        '. '代表任意字符
        '* '代表 0-正无穷
        .* 贪婪匹配 匹配所有被人
        $结尾
    m.groupdict()['postcolon'].split('.') 取 postcolon 组的 字典按。号分割
"""

print('分句')
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

# 让我们看看前几个：
print(sentences_strings_ted[:5])
del input_text, input_text_noparens
"""
删除输入文本，输入文本无 # 如果需要保存一些RAM，请取消注释：这些字符串大约为50MB。
既然我们有了句子，我们就可以把它们标记成单词了。
当然，这种标记化是不完美的。例如，有多少标记是“can't”，我们在哪里/如何拆分它？
我们将采用最简单的简单方法在空间上拆分。
在拆分之前，我们删除非字母数字字符，如标点符号。
"""
print('分词')
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    # tokens = re.sub(r'[{}]+'.format('-!:,.;?"'), " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

# Two sample processed sentences:
print(sentences_ted[0])
print(sentences_ted[1])

print('Part 2: 词频统计')
# 如果将前1000个单词的计数存储在名为“counts_ted_top1000”的列表中，下面的代码将绘制writeup中请求的直方图。
# 方法1 使用 from collections import Counter
sentences_ted_all = [i for row in sentences_ted for i in row]
c = Counter(sentences_ted_all)
counts_ted_top1000_counter = c.most_common(1000)
print(counts_ted_top1000_counter[:5])

# 方法2 使用 from sklearn.feature_extraction.text import CountVectorizer
# argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引) -X则取反
vec = CountVectorizer()
X = vec.fit_transform(sentences_strings_ted)  # 不同点不需要分词 直接用句子处理
counts_ted_top1000_index = np.array(np.argsort(-X.sum(axis=0))).squeeze()[:1000]
counts_ted_top1000_word = np.array(vec.get_feature_names())[counts_ted_top1000_index]
counts_ted_top1000 = np.array(-1 * np.sort(-X.sum(axis=0))).squeeze()[:1000]


counts_ted_top40_index = np.array(np.argsort(-X.sum(axis=0))).squeeze()[:40]        # 前40索引
counts_ted_top40_word = np.array(vec.get_feature_names())[counts_ted_top40_index]   # 前40 word
counts_ted_top40 = np.array(-1 * np.sort(-X.sum(axis=0))).squeeze()[:40]            # 前40 计数
#
print(counts_ted_top40_word)


# Plot distribution of top-1000 words
#
print('绘制直方图')
hist, edges = np.histogram(counts_ted_top1000, density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Top-1000 words distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)

print('Part 2: 训练Word2Vec')
"""
· sentences：切分句子的列表。
· size：嵌入向量的维数
· window：你正在查看的上下文单词数
· min_count：告诉模型忽略总计数小于这个数字的单词。
· workers：正在使用的线程数
· sg：是否使用skip-gram或CBOW
"""
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=10, workers=4, sg=0)
dictionary_ted = corpora.Dictionary(sentences_ted)
print(len(dictionary_ted), dictionary_ted)

print('训练FastText')
model_fas = FastText(sentences=sentences_ted, size=100, window=5, min_count=10, workers=4, sg=1)

print('Part 4: Ted Learnt Representations')
# Finding similar words: (see gensim docs for more functionality of `most_similar`)
# 查找哪些词与“man”这个词最相似。
man = model_ted.wv.most_similar('man')
computer = model_ted.wv.most_similar("computer")
print('man', '——>', man)
print('computer', '——>', computer)
man2 = model_fas.wv.most_similar('man')
Gastroenteritis = model_fas.wv.most_similar("Gastroenteritis")

# 补充作业计算两个词向量的余弦距离，-》1则更接近


def CosineDistance(vv1, vv2):  # 计算夹角余弦
    return np.dot(vv1, vv2) / (np.linalg.norm(vv1) * np.linalg.norm(vv2))


print('夹角余弦')
print(CosineDistance(model_ted.wv['man'], model_ted.wv['kid']))
print(CosineDistance(model_ted.wv['computer'], model_ted.wv['kid']))

# #### t-SNE visualization
# 低维空间下，使用t分布替代高斯分布表达两点之间的相似度
words_top_ted = counts_ted_top1000_word
words_top_vec_ted = model_ted[words_top_ted]


tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(words_top_vec_ted)


p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:, 0],
                                    x2=words_top_ted_tsne[:, 1],
                                    names=words_top_ted))

p.scatter(x="x1", y="x2", size=8, source=source)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(labels)
show(p)

# ### Part 5: Wiki Learnt Representations

print('-----------下载维基百科文本数据-----------')
if not os.path.isfile(r'D:\C\NLP\Data\wikitext-103-raw-v1.zip'):
    urllib.request.urlretrieve("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
                               filename="wikitext-103-raw-v1.zip")

# In[]
print('-----------解压并读取数据-----------')
with zipfile.ZipFile(r'D:\C\NLP\Data\wikitext-103-raw-v1.zip', 'r') as z:
    input_text = str(z.open('wikitext-103-raw/wiki.train.raw', 'r').read(), encoding='utf-8')

print('-----------分句-----------')
sentences_wiki = []
for line in input_text.split('\n'):
    s = [x for x in line.split('.') if x and len(x.split()) >= 5]
    sentences_wiki.extend(s)

for s_i in range(len(sentences_wiki)):
    sentences_wiki[s_i] = re.sub("[^a-z0-9]", " ", sentences_wiki[s_i].lower())
    sentences_wiki[s_i] = re.sub(r'\([^)]*\)', '', sentences_wiki[s_i])
del input_text


print('-----------取1/5的数据-----------')
# sample 1/5 of the data  shuffle打乱顺序
shuffle(sentences_wiki)
print(len(sentences_wiki))
sentences_wiki = sentences_wiki[:int(len(sentences_wiki) / 2)]
print(len(sentences_wiki))

wiki_ted = []
for wiki_str in sentences_wiki:
    tokens = re.sub(r"[^a-z0-9]+", " ", wiki_str.lower()).split()
    # tokens = re.sub(r'[{}]+'.format('-!:,.;?"'), " ", sent_str.lower()).split()
    wiki_ted.append(tokens)

model_wiki = Word2Vec(sentences=wiki_ted, size=100, window=5, min_count=10, workers=4, sg=0)
dictionary_wiki = corpora.Dictionary(wiki_ted)
print(len(dictionary_wiki), dictionary_wiki)
# #### t-SNE visualization

wiki = CountVectorizer()
Y = wiki.fit_transform(sentences_wiki)
wiki_ted_top1000_index = np.array(np.argsort(-Y.sum(axis=0))).squeeze()[:1000]
wiki_ted_top1000_word = np.array(wiki.get_feature_names())[wiki_ted_top1000_index]
wiki_ted_top1000 = np.array(-1 * np.sort(-Y.sum(axis=0))).squeeze()[:1000]

words_top_wiki = wiki_ted_top1000_word
# This assumes words_top_wiki is a list of strings, the top 1000 words
words_top_vec_wiki = model_wiki[words_top_wiki]

tsne = TSNE(n_components=2, random_state=0)
words_top_wiki_tsne = tsne.fit_transform(words_top_vec_wiki)


p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

source = ColumnDataSource(data=dict(x1=words_top_wiki_tsne[:, 0],
                                    x2=words_top_wiki_tsne[:, 1],
                                    names=words_top_wiki))

p.scatter(x="x1", y="x2", size=8, source=source)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(labels)

show(p)


clustering = KMeans(n_clusters=100)  # 聚几个类

clustering.fit(words_top_vec_wiki)
clu_labels = clustering.labels_  # 0-100 个标签

print(wiki_ted_top1000_word[clu_labels == 0])
print(wiki_ted_top1000_word[clu_labels == 1])

colors = [
    '#%d%d%d' % (int(2.55 * r), int(1.5 * r), int(1 * r)) for r in clu_labels
]
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

clustering_source = ColumnDataSource(data=dict(x1=words_top_wiki_tsne[:, 0],
                                               x2=words_top_wiki_tsne[:, 1],
                                               names=words_top_wiki,
                                               colors=colors))

p.scatter(x="x1", y="x2", size=8, source=clustering_source, fill_color='colors')

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=clustering_source, text_align='center')
p.add_layout(labels)

show(p)
