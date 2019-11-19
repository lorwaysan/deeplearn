import numpy
import scipy.special
import csv
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from numpy import *


class deepLearn:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        #随机生成初始权重
        self.wih = numpy.random.normal(0.0 , pow(self.hnodes , -0.5),(self.hnodes , self.inodes))
        self.who = numpy.random.normal(0.0 , pow(self.onodes , -0.5),(self.onodes , self.hnodes))
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def train(self,inputs_list,targets_list):
        #将输入数据进行T变换
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih,inputs)   
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        #errors_hidden = weights^T_hidden_output * errors_output
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #隐藏层和输出层权重更新
        #ΔWj,k = α * Ek * sigmod(Ok) * (1 - sigmod(Ok)) * Oj^T
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        #输入层和隐藏层权重更新
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)   
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

#设置输入层隐藏层输出层
input_nodes = 13
hidden_nodes = 1200
output_nodes = 10

#设置学习率
learning_rate = 0.1
n = deepLearn(input_nodes,hidden_nodes,output_nodes,learning_rate)
#pd读取数据
train = pd.read_csv('train.csv', keep_default_na=True,na_values='?',header=None)
test = pd.read_csv('test.csv', keep_default_na=True,na_values='?',header=None)

#将数据转化成numpy数组
train=numpy.array(train)
test=numpy.array(test)

#标准化处理
train_scaled = preprocessing.scale(train)
test_scaled = preprocessing.scale(test)

#补充缺失值
imputer = Imputer(missing_values=numpy.nan, strategy="mean", axis=0)    
train=imputer.fit_transform(train)
test=imputer.fit_transform(test)

#转化为python数组
training_data_list = train.tolist()
test_data_list = test.tolist()

epochs = 10 #循环次数
for e in range(epochs):
    for all_values in training_data_list:
        inputs = all_values[0:2]
        inputs = (numpy.asfarray(all_values[0:13]) / 10.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[13])] = 0.99
        n.train(inputs,targets)
        pass
    pass



result = []
count = 0
#测试
for all_values in test_data_list:
    inputs = (numpy.asfarray(all_values[0:13]) / 10.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    count += 1
    result1 = []
    result1.append(count)
    result1.append(label)
    result.append(result1)
    
    pass

#写入csv文件
with open('123.csv','w') as jg:
    writer = csv.writer(jg,lineterminator='\n')
    writer.writerow(['id','y'])
    for item in result:
        writer.writerow(item)

