from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


max_features = 10000    # 原始特征词的个数（原始具有10000个维度的特征，每个特征相互独立）
embbeding_dim = 8             # 将每个原始特征词用一个8维向量表示（输出具有8个维度的特征）
maxlen = 20             # 每句话仅取20个词（均属于10000个特征词）

# 加载数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)

# 将样本数据整形为20的长度
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 构建网络，训练embbeding层
model = Sequential()
model.add(Embedding(max_features, embbeding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
