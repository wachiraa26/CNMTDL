import numpy as np
from PIL import Image
import os
import copy
import time
import cv2
import sys
import h5py
from data import get_2d_data
from Hodge_2d import get_D1 as get_D1_2d
from Hodge_2d import get_D0 as get_D0_2d
from Hodge_2d import get_BIG_decomposition as decomposition_2d



def remove_bg_2d(data):
    import copy
    data2 = copy.deepcopy(data)
    data2[data2==0] = 300
    return data2
        
def get_omega_2d(img,size):
    data = np.array(img)/255.0
    data = data.reshape(size,size,-1)
    res = []
    for i in range(data.shape[2]):
        data_now = data[:,:,i]
        data_now = np.pad(data_now, pad_width=3, mode='constant', constant_values=0)
        Gx = np.zeros_like(data_now)  
        Gy = np.zeros_like(data_now)
        Gx[1:-1, 1:-1] = (data_now[1:-1, 2:] - data_now[1:-1, :-2]) / 2
        Gy[1:-1, 1:-1] = (data_now[2:, 1:-1] - data_now[:-2, 1:-1]) / 2
    
        X = Gx[1:-1, 1:-1]
        Y = Gy[1:-1, 1:-1]
        H = (X[:, :-1] + X[:, 1:]) / 2
        V = (Y[:-1, :] + Y[1:, :]) / 2
        H = H.reshape(-1,1)
        V = V.reshape(-1,1)
        vector = np.vstack((H,V))
        res.append(vector)
    return res
    
def get_2d_decomposition(dataname,start,end,size=224):
    folder = './' + dataname + '/' 
    all_data = get_2d_data(dataname,size)
    print(len(all_data))
    
    for i in range(start,end):
        img = all_data[i][0]
        omega = get_omega_2d(img,size)
        data = np.array(img).reshape(size,size,-1)
        label = np.array(all_data[i][1])
        
        with h5py.File(folder + str(i) + '.h5', 'w') as file:
            file.create_dataset('label', data=label)
        
            for k in range(len(omega)):
                M = size + 4
                N = size + 4
                data2 = np.ones((M,N))*300
                data2[2:2+size,2:2+size] = data[:,:,k]
                manifold = remove_bg_2d(data2)
                D1 = get_D1_2d(M,N)
                D0 = get_D0_2d(M,N)
                v = 256
                vector_all, vector_cur, vector_div, vector_har = decomposition_2d(omega[k],D0,D1,M,N,manifold,v)
                
                name = str(i) + '-' + str(v) + '-' + str(k) + '-cur'
                file.create_dataset(name,data=vector_cur)
                
                name = str(i) + '-' + str(v) + '-' + str(k) + '-div'
                file.create_dataset(name,data=vector_div)
                
                name = str(i) + '-' + str(v) + '-' + str(k) + '-har'
                file.create_dataset(name,data=vector_har)
                
                print(i,k,'ok')
        
