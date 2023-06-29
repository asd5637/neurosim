# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:52:08 2023

@author: A90230
"""


import numpy as np
import math

DIR = "C:\\Users\\A90230\\Desktop\\dat files_CIM\\"
    
def ReadDat(filepath, isWeight=False):
    data = []
    with open(filepath) as dat:
        data.append([filedata.split('\n')[0] for filedata in dat])
        
    out = []
    for d in data[0]:
        if d == '': continue
        
        tmp = []
        
        for val in d:
            v = int(val, 16)
            if isWeight:
                tmp.append(v-16) if v > 7 else tmp.append(v)
            else:
                tmp.append(v)
        
        out.append(tmp)
        
    return out

#%%
class Conv:
    def __init__(self, IFM, weight, stride):
        self.IFM = IFM
        self.weight = weight
        self.stride = stride
            
    def padding(self):
        img_size = int(math.sqrt(np.shape(self.IFM)[0]))
        channel = np.shape(self.IFM)[1]
        IFM = np.reshape(self.IFM, (img_size, img_size, channel))
        IFM = np.pad(IFM, ((1, 1), (1, 1), (0, 0)))
        
        return IFM
    
    def relu(self, OFM):
        OFM[OFM < 0] = 0
        OFM[OFM > 127] = 127
        OFM = OFM // 8
        
        return OFM
        
    def compute(self):
        IFM = self.padding()
        
        img_size = int(math.sqrt(np.shape(self.IFM)[0]))
        channel = np.shape(self.IFM)[1]
        if self.stride == 1:
            OFM = np.zeros((img_size, img_size, channel))
        else:
            OFM = np.zeros((img_size // 2, img_size //2, channel))
        
        row = np.shape(IFM)[0]
        col = np.shape(IFM)[1]
        
        for k in range(0, 16, 2):                   # 16 kernel : take two of them each time
            for n in range(9):                      # 3x3 kernel
                w1 = self.weight[9 * k + n]
                w2 = self.weight[9 * (k+1) + n]
                
                k_row, k_col = n // 3, n % 3
                
                cur_row, cur_col = -1, -1
                for r in range(0 + k_row, row - (3 - k_row) + 1, self.stride):
                    cur_row += 1
                    for c in range(0 + k_col, col - (3 - k_col) + 1, self.stride):
                        curInput = IFM[r][c]
                        cur_col += 1
                        
                        out1 = sum(np.multiply(curInput, w1))
                        out2 = sum(np.multiply(curInput, w2))
                        
                        OFM[cur_row][cur_col][k] += out1
                        OFM[cur_row][cur_col][k+1] += out2
                        
                    cur_col = -1
                    
        OFM =  self.relu(OFM)
                        
        return OFM
    
#%%
layers = 6
ALL_W = []
ALL_IFM = []
IMG_DIR = "image3\\"
ALL_IFM.append(ReadDat(DIR + IMG_DIR + "input.dat"))
for layer in range(1, layers + 1):
    w = ReadDat(DIR + "conv" + str(layer) + "_w.dat", True)
    ifm = ReadDat(DIR + IMG_DIR + "conv" + str(layer) + "_out.dat")
    
    ALL_W.append(w)
    ALL_IFM.append(ifm)
    
#%%
conv1 = Conv(ALL_IFM[0], ALL_W[0], 1)
conv2 = Conv(ALL_IFM[1], ALL_W[1], 1)
conv3 = Conv(ALL_IFM[2], ALL_W[2], 2)
conv4 = Conv(ALL_IFM[3], ALL_W[3], 1)
conv5 = Conv(ALL_IFM[4], ALL_W[4], 2)
conv6 = Conv(ALL_IFM[5], ALL_W[5], 2)

#%%
model = dict()
model[0] = conv1
model[1] = conv2
model[2] = conv3
model[3] = conv4
model[4] = conv5
model[5] = conv6

#%%
img_size = int(math.sqrt(np.shape(ALL_IFM[0])[0]))

for layer in range(layers):
    out = model[layer].compute()
    
    img_size = int(img_size / model[layer].stride)
    ans = np.reshape(ALL_IFM[layer + 1], (img_size, img_size, 16))
    
    print((out == ans).all())
    
