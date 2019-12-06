import os
from PIL import Image
import math
import csv
import numpy
def convert_image(x):
    image = Image.open(x)
    image = image.convert('L')  #转化为灰度图
    threshold = 127             #设定的二值化阈值
    table = []                  #table是设定的一个表，下面的for循环可以理解为一个规则，小于阈值的，就设定为0，大于阈值的，就设定为1
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    image = image.point(table,'1')  

def cut_image(image):
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 5):
            box = (30*i,0,14+30+i*30,30)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def buildvector(image):
    """
    图片转换成矢量,将二维的图片转为一维
    """
    result = {}
    count = 0
    for i in image.getdata():
        result[count] = i
        count += 1
    return result

class CaptchaRecognize:
    def __init__(self):
        self.letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v','w','x','y','z','AA','BB', 'CC', 'DD', 'EE', 'FF','GG','HH','II','JJ','KK','LL', 'MM', 'NN', 'OO', 'PP','QQ','RR','SS','TT','UU','VV', 'WW', 'XX', 'YY', 'ZZ']
        self.loadSet()

    def loadSet(self):
        self.imgset = []
        for letter in self.letters:
            temp = []
            for img in os.listdir('./qiege/%s'%(letter)):
                # 将图片转成一维向量，放入temp列表中
                temp.append(buildvector(Image.open('./qiege/%s/%s'%(letter[0],img))))
            # 标签与对应图片转换成的向量，以字典形式存到imgset 如：letter为1,temp就是1文件夹下图片的向量
            self.imgset.append({letter[0]:temp})

    def magnitude(self,concordance):
        """
        利用公式求计算矢量大小
        """
        total = 0
        for word,count in concordance.items():
            # count 为向量各个单位的值
            total += count ** 2
        return math.sqrt(total)

    def relation(self, concordance1, concordance2):
        """
        计算矢量之间的 cos 值
        """
        relevance = 0
        topvalue = 0
        # 遍历concordance1向量，word 当前位置的索引，count为值
        for word, count in concordance1.items():
            # 当concordance2有word才继续，防止索引超限
            if word in concordance2:
                #print(type(topvalue), topvalue, count, concordance2[word])
                topvalue += count * concordance2[word]
                #time.sleep(10)
        return topvalue / (self.magnitude(concordance1) * self.magnitude(concordance2))

    def recognise(self,image):
        """
        识别验证码
        """
        # 二值化,将图片按灰度转为01矩阵
        image = convert_image(image)
        # 对完整的验证码进行切割，得到字符图片
        images = cut_image(image)
        vectors = []
        for img in images:
            vectors.append(buildvector(img))  # 将字符图片转一维向量
        result = []
        for vector in vectors:
            guess=[]
            # 让字符图片和训练集逐一比对
            for image in self.imgset:
                for letter,temp in image.items():
                    relevance=0
                    num=0
                    # 遍历一个标签下的所有图片
                    for img in temp:
                        # 计算相似度
                        relevance+=self.relation(vector,img)
                        print (vector,img)
                        num+=1
                    # 求出相似度平均值
                    relevance=relevance/num
                    guess.append((relevance,letter))
            # 对cos值进行排序，cos值代表相识度
            guess.sort(reverse=True)
            result.append(guess[0])  #取最相似的letter，作为该字符图片的值
        return result

if __name__ == '__main__':
    imageRecognize=CaptchaRecognize()
    lt1 = os.listdir('./test/')         #读取测试集列表
    i = 0
    for img in lt1:
        image = Image.open('./test/'+img)       #打开测试集中的测试图片
        result = imageRecognize.recognise(image)
        string = [''.join(item[1]) for item in result]
        lt2 = [i,result]
        with open('123.csv','w') as t:    #写入csv
            csv_write = csv.writer(t)
            csv_write.writerow(lt2)
        i += 1

        