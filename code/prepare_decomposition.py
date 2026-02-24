import os
from decomposition import get_2d_decomposition

data_2d = [ 'retinamnist','bloodmnist', 'organamnist', 'organcmnist', 'organsmnist']

def decomposition_to_file(dataname):
    data_num = { 'retinamnist':1600, 'bloodmnist':17092, 'organamnist':58830, 'organcmnist':23583, 'organsmnist':25211}
    
    folder = './' + dataname
    if not os.path.exists(folder):
        os.makedirs(folder)
    if dataname in data_2d:
        get_2d_decomposition(dataname,0,data_num[dataname])


dataname = data_2d[0] 
decomposition_to_file(dataname)
