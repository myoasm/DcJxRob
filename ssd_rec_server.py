
# coding: utf-8

# In[1]:


get_ipython().system('pwd')


# In[ ]:


#  author : Dingchao
#  data : 20190526
#  单目标位置识别的服务脚本


import os
import d2lzh as d2l

from mxnet import gluon, image
from mxnet.gluon import  utils as gutils
from mxnet import contrib, image, nd
from random import shuffle
import subprocess
import mxnet as mx
import numpy as np
from  mxnet import  autograd, gluon,  image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
import time, random
from random import shuffle
import cv2
import  d2lzh as d2l
from mxnet import  autograd, contrib, gluon, image, init, nd
from mxnet.gluon import  loss as gloss, nn
import time
from socket import *

obj = cv2.imread('/home/dingchao/objs/smll3.jpg')
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],  [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1



W=obj.shape[1]
H=obj.shape[0]
def get_blk(i):
    if i==0:
        blk=base_net()
    elif i==4:
        blk=nn.GlobalMaxPool2D()
    else:
        blk=down_sample_blk(128)
    return blk

def base_net():
    blk=nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

def down_sample_blk(num_channels):
    blk=nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
               nn.BatchNorm(in_channels=num_channels),
               nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors*(num_classes+1), kernel_size=3, padding=1)

def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors*4, kernel_size=3, padding=1)

def flatten_pred(pred):
    return pred.transpose((0,2,3,1)).flatten()

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y=blk(X)
    anchors=contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds=cls_predictor(Y)
    bbox_preds=bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

def concat_preds(preds):
     return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors,
                                                      num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape(
                    (0, -1, self.num_classes + 1)), concat_preds(bbox_preds))
    

def view_bar(num, total):
    rate = num / total                        #得到现在的比率，0<rate<1
    rate_num = int(rate * 100)                #将比率百分化，0<rate_num<100
    r = '\r[%s%s]' % (">"*num, " "*(100-num)) #进度条封装
    sys.stdout.write(r)                       #显示进度条
    sys.stdout.write(str(num)+'%')            #显示进度百分比
    sys.stdout.flush()  


def predict(X):
        anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
        cls_probs = cls_preds.softmax().transpose((0, 2, 1))
        output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
        return output[0, idx]


def display(img, output, threshold):
    #fig = d2l.plt.imshow(img.asnumpy())
    i = 0
    for row in output:
        score = row[1].asscalar()

        if score < threshold:
            continue
        i += 1
        print("There is an obj, num %d, position as follows:" % i)
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        point1 = (int(row[2].asscalar() * w), int(row[3].asscalar() * h))
        point2 = (int(row[4].asscalar() * w), int(row[5].asscalar() * h))
        the_center_point = (int(160 + (point1[0] + point2[0]) / 4), int(120 + (point1[1] + point2[1]) / 4))
        zifuchuan = str(the_center_point[0]) +" " + str(the_center_point[1])
        print(the_center_point)
        print(zifuchuan)
        
        tcpCliSock1 = socket(AF_INET, SOCK_STREAM)
        print('返回坐标')
        tcpCliSock1.connect((str(addr[0]), 8080))
        tcpCliSock1.send(zifuchuan.encode())
        print("坐标已返回")
       # d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


net = TinySSD(num_classes=1)
net.initialize()
X = nd.zeros((32, 3,200,255))
anchors, cls_preds, bbox_preds = net(X)
batch_size, edge_size = 32, 255
net.load_params("test.wellsave")


# # socket 服务
HOST = ''  #对bind（）方法的标识，表示可以使用任何可用的地址
PORT = 8080  #设置端口
BUFSIZ = 1024  #设置缓存区的大小
ADDR = (HOST, PORT)
count = 1
#print("正在等待客户端连接")


while True:
    print('等待第%d张图片的到来...'%(count))
    tcpSerSock = socket(AF_INET, SOCK_STREAM) 
    tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)#使我们的端口可以重新利用 
    tcpSerSock.bind(ADDR)  # 绑定地址
    tcpSerSock.listen(5)
    tcpCliSock, addr = tcpSerSock.accept()
    #     while True:
    received_size = 0
    f = open("needtorec.jpg", "wb")
    count1 = 0
# #用c#时的方式
    while True:  # and  timeToolong == 0:
        data = tcpCliSock.recv(BUFSIZ)
        #f = open("needtorec.jpg", "wb")

        received_size += len(data)
        i = int(received_size//1024//121*100)
        view_bar(i, 100)
        #time.sleep(0.3)
        if data.endswith(b'woaini'):
            break
        f.write(data)
        count1 += 1
        
#python测试
    # filesize = int(tcpCliSock.recv(BUFSIZ).decode())
    # print("filesize is ", filesize)
    # tcpCliSock.send("get it!".encode())
    # f = open("needtorec.jpg", "wb")
    # received_size = 0
    # while received_size < filesize:
    #     data = tcpCliSock.recv(BUFSIZ)
    #     f.write(data)
    #     received_size += len(data)
    #     print("已接收:", received_size // 1024, "kb")
    #     # if data == "woaini":
    #     # break
    print("\n图片接收完毕...","图片大小为%dKB"%(received_size // 1024))
    f.close()
    tcpCliSock.close()
    tcpSerSock.close()
    

    ctx = mx.cpu()
    edge_size = 255
    # img = image.imread('needtorec.jpg')
    #img1 = cv2.imread('/home/dingchao/1111.jpg')
    img1 = cv2.imread('needtorec.jpg')
    imgmulti2 = np.zeros((240, 320, 3), np.uint8)
    imgmulti2[:, :, 0] = img1[121:361, 161:481, 0]
    imgmulti2[:, :, 1] = img1[121:361, 161:481, 1]
    imgmulti2[:, :, 2] = img1[121:361, 161:481, 2]
    imgmulti2 = cv2.resize(imgmulti2, (640, 480))
    # net = TinySSD(num_classes=1)
    # ctx, net = d2l.try_gpu(), TinySSD(num_classes=1)
    cv2.imwrite("muti2.jpg", imgmulti2)
    img = image.imread('./muti2.jpg')
    # img = image.imresize(img,640*3,480*3)
    print(img.shape)
    # img = image.imread(os.path.join(data_dir,'12.jpg'))
    feature = image.imresize(img, edge_size, edge_size).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
    output = predict(X)
    d2l.set_figsize((5, 5))
    big = 0
    scorelist = []
    for row in output:
        score = row[1].asscalar()
        scorelist.append (score)
        if score > big:
            big = score
            print(big)
    threshold = big
    print(threshold)
    display(img, output, threshold)
    count += 1
    


# In[2]:


get_ipython().system('lsof -i:8080')
    


# In[ ]:


get_ipython().system('kill -9  20628')


# In[3]:


scorelist


# In[5]:


scorelist[1
         ]

