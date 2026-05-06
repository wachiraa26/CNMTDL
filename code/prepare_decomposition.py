import os
from decomposition import get_2d_decomposition,get_3d_decomposition







data_2d = [ 'retinamnist', 'pneumoniamnist', 'dermamnist', 'bloodmnist', 'organamnist', 'organcmnist', 'organsmnist', 'pathmnist', 'octmnist', 'chestmnist', 'tissuemnist' ]
data_3d = [ 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d' ]


def decomposition_to_file(dataname):
    data_num = { 'retinamnist':1600, 'pneumoniamnist':5856, 'dermamnist':10015, 'bloodmnist':17092, 'organamnist':58830, 'organcmnist':23583, 'organsmnist':25211, 'pathmnist':107180,
                 'octmnist':109309, 'chestmnist':112120, 'tissuemnist':236386, 'organmnist3d':1742, 'nodulemnist3d':1633, 'adrenalmnist3d':1584, 'fracturemnist3d':1370, 
                 'vesselmnist3d':1908, 'synapsemnist3d':1759  }
    
    folder = './' + dataname
    if not os.path.exists(folder):
        os.makedirs(folder)
    if dataname in data_2d:
        get_2d_decomposition(dataname,0,data_num[dataname])
    elif dataname in data_3d:
        get_3d_decomposition(dataname,0,data_num[dataname])


dataname = data_2d[4] 
#dataname = data_3d[1]   
decomposition_to_file(dataname)
