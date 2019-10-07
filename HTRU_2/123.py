import numpy
import scipy.special
import csv

class deepLearn:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.normal(0.0 , pow(self.hnodes , -0.5),(self.hnodes , self.inodes))
        self.who = numpy.random.normal(0.0 , pow(self.onodes , -0.5),(self.onodes , self.hnodes))
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih,inputs)   
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)   
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


input_nodes = 2
hidden_nodes = 100
output_nodes = 2

learning_rate = 0.15
n = deepLearn(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("HTRU_2_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 100
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = all_values[0:2]
        inputs = (numpy.asfarray(all_values[0:2]) / 200.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[2])] = 0.99
        n.train(inputs,targets)
        pass
    pass

test_data_file = open("HTRU_2_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

result = []
count = 0
for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[0:2]) / 200.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    count += 1
    result1 = []
    result1.append(count)
    result1.append(label)
    result.append(result1)
    
    pass


with open('123.csv','w') as jg:
    writer = csv.writer(jg,lineterminator='\n')
    writer.writerow(['id','y'])
    for item in result:
        writer.writerow(item)
