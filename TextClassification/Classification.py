# -*- coding:utf-8 -*-
# @Time    : 2019/10/14 10:26
# @Author  : Ray.X
# In[] 读取文本
import zipfile
import lxml.etree
"""
    获取xml中的有效文本 content 数据 keywords 标签
"""
with zipfile.ZipFile(r'D:\C\NLP\Data\ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
content = (doc.xpath('//content/text()'))   # 获取<content>下的文本 数组
keywords = (doc.xpath('//keywords/text()'))  # 获取<keywords[i]s>下的文本 标签
z.close()
del doc
# In[] 处理文本
"""
    处理content 去标点符号 分词 去停用词
"""
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

words_list = [word_tokenize(re.sub(r'[^\w\s]', '', str(content[i].lower()))) for i in range(len(content))]
stopwords = stopwords.words('english')
clean_words = []
clean_content = []
for i in range(len(words_list)):
    for word in words_list[i]:
        if word not in stopwords:
            clean_words.append(word)
    clean_content.append(' '.join(clean_words))
    clean_words = []

# In[] 处理标签

"""
    处理keywords
    - None of the keywords → ooo
    - “Technology” → Too
    - “Entertainment” → oEo
    - “Design” → ooD
    - “Technology” and “Entertainment” → TEo
    - “Technology” and “Design” → ToD
    - “Entertainment” and “Design” → oED
    - “Technology” and “Entertainment” and “Design” → TED
"""
for i in range(len(keywords)):
    if 'technology' not in keywords[i] and 'entertainment'not in keywords[i] and 'design' not in keywords[i]:
        keywords[i] = 'ooo'
    elif 'technology' in keywords[i] and 'entertainment' not in keywords[i] and 'design' not in keywords[i]:
        keywords[i] = 'Too'
    elif 'technology' not in keywords[i] and 'entertainment' in keywords[i] and 'design' not in keywords[i]:
        keywords[i] = 'oEo'
    elif 'technology' not in keywords[i] and 'entertainment' not in keywords[i] and 'design' in keywords[i]:
        keywords[i] = 'ooD'
    elif 'technology' in keywords[i] and 'entertainment' in keywords[i] and 'design' not in keywords[i]:
        keywords[i] = 'TEo'
    elif 'technology' in keywords[i] and 'entertainment' not in keywords[i] and 'design' in keywords[i]:
        keywords[i] = 'ToD'
    elif 'technology' not in keywords[i] and 'entertainment' in keywords[i] and 'design' in keywords[i]:
        keywords[i] = 'oED'
    else:
        keywords[i] = 'TED'
# In[] 整理文本与标签
import pandas as pd
"""
    把处理后的keywords 与 文本数据组合
"""
df_data = {'keywords': keywords, 'content': clean_content}
df = pd.DataFrame(df_data)
"""
将文本与量化后的keywords组合到DataFame
    df[''].factorize()可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字。
    .drop_duplicates()去除重复项
    .sort_values 排序
    pandas sort_values 排序后，index 也发生了改变，不改变的情况下需要 reset_index(drop = True)
    key_id_df keywords量化表 后期作标签还原
"""
df['key_id'] = df['keywords'].factorize()[0]
key_id_df = df[['keywords', 'key_id']].drop_duplicates().sort_values('key_id').reset_index(drop=True)
df.sample(10)

# In[] 简单的标签分布统计
from matplotlib import pyplot as plt
from collections import Counter
"""
    做个简单的标签统计
"""
c = Counter(keywords)
d = {'keywords': list(c.keys()), 'counts': list(c.values())}
count_key = pd.DataFrame(d)
count_key.plot(x='keywords', y='counts', kind='bar', legend=False, figsize=(8, 5))
plt.title("类目分布")
plt.ylabel('counts', fontsize=18)
plt.xlabel('keywords', fontsize=18)
plt.show()
plt.close()

# In[] 释放缓存
del content, words_list,  clean_words, keywords, clean_content, df_data

# In[] 向量化文本
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
"""
    我们要将content数据进行向量化处理,我们要将每条content转换成一个整数序列的向量
    设置最频繁使用的50000个词
    设置每条content最大的词语数为250个(超过的将会被截去,不足的将会被补0)
"""
# 设置最频繁使用的50000个词
MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['content'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))

X = tokenizer.texts_to_sequences(df['content'].values)
# 填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# 多类标签的onehot展开
Y = pd.get_dummies(df['key_id']).values
print(X.shape)
print(Y.shape)
# In[] 拆分训练集和测试集
"""
    拆分1585组训练 250组验证 250 组测试
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=250, random_state=1)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=250, random_state=1)

print(X_train.shape, Y_train.shape)
print(X_validation.shape, Y_validation.shape)
print(X_test.shape, Y_test.shape)
# In[] 定义模型
"""
    定义模型
"""
from keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import Dense
from keras.layers import LSTM


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(8, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# In[]
from keras.callbacks import EarlyStopping
"""
    训练
"""
epochs = 5
batch_size = 64
# validation_split=0.1 自动切分验证集
# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#  validation_data=(X_test,y_test) 手动切分的数据集
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,  validation_data=(X_validation, Y_validation),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
# 绘制loss图
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 绘制acc图
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
Y_test = Y_test.argmax(axis=1)

# 生成混淆矩阵
conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=key_id_df.keywords.values, yticklabels=key_id_df.keywords.values)
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)

from sklearn.metrics import classification_report

print('accuracy %s' % accuracy_score(y_pred, Y_test))
print(classification_report(Y_test, y_pred, target_names=key_id_df['keywords'].values))


def predict(text):
    """
    测试函数
    :param text:
    :return:
    """
    txt = re.sub(r'[^\w\s]', '', text.lower())
    txt = [" ".join([w for w in word_tokenize(txt) if w not in  stopwords.words('english')])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    key_id= pred.argmax(axis=1)[0]
    return key_id_df[key_id_df.key_id==key_id]['keywords'].values[0]

predict("So far we've been looking to this new piece of mechanical technology or that great next generation robot as part of a lineup to ensure our species safe passage in space. Wonderful as they are, I believe the time has come for us to complement these bulky electronic giants with what nature has already invented: the microbe, a single-celled organism that is itself a self-generating, self-replenishing, living machine.")