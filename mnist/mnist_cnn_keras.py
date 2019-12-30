from keras.utils import to_categorical
from keras import layers
from keras import models
import numpy as np


model = models.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#flatten为展平一个张量
model.add(layers.Flatten())
#dense添加全连接层
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


with open('./mnist_train.csv') as t1:
    train = []
    train_0 = t1.readlines()
    train_2 = []
    train_target = []
    for i in train_0:
        i = i.replace('\n','')
        train_1 = i.split(',')
        train_2.append(train_1)
    for i in train_2:
        traintgt = i[0]
        train_target.append(int(traintgt))
        train1 = i[1:]
        train.append(train1)
    train = np.array(train)
    train_images = np.reshape(train,(60000,28,28,1))
    train_images = train_images.astype('float32') / 255

with open('./mnist_test.csv') as t2:
    test = []
    test_0 = t2.readlines()
    test_2 = []
    test_target = []
    for i in test_0:
        i = i.replace('\n','')
        test_1 = i.split(',')
        test_2.append(test_1)
    for i in test_2:
        testtgt = i[0]
        test_target.append(int(testtgt))
        test1 = i[1:]
        test.append(test1)
    test = np.array(test)
    test_images = np.reshape(test,(10000,28,28,1))
    test_images = test_images.astype('float32') / 255

train_target = np.array(train_target)
test_target = np.array(test_target)
train_labels = to_categorical(train_target)
test_labels = to_categorical(test_target)


#print(test_labels)

model.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10,shuffle=True)
xx = model.evaluate(test_images,test_labels)
print(xx)
