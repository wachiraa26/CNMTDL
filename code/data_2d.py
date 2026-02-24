import numpy as np
import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
from medmnist.dataset import MedMNIST2D






       
def load_decomposition_img(dataname,typ,index,label):
    train_test_num = {'retinamnist':[1080,400],'bloodmnist':[11959,3421],'organamnist':[34561,17778],'organcmnist':[12975,8216],'organsmnist':[13932,8827]}
    train_num,test_num = train_test_num[dataname]
    channel_num = {'retinamnist':3,'bloodmnist':3,'organamnist':1,'organcmnist':1,'organsmnist':1}
    
    if channel_num[dataname]==3:
        rgb_list = [0,1,2]
    else:
        rgb_list = [0]
    size = 224
    if typ=='train':
        now_index = index
    elif typ=='test':
        now_index = index+train_num
    elif typ=='val':
        now_index = index+train_num+test_num
        
    filename = './' + dataname + '/' + str(now_index) + '.h5'
    with h5py.File(filename, 'r') as file:
        label1 = file['label'][:]
        assert label1==label
        
        
        img_high = []
        for k in rgb_list:
            for typ in ['cur','div','har']: 
                v = 256
                name = str(now_index) + '-' + str(v) + '-' + str(k) + '-' + typ  # cur, div, har
                img_now1 = np.round(file[name][:],8)
                img_high.append(img_now1[:,1:1+size,1:1+size])
        
        img_high = np.vstack(img_high)*2.0
        
        return torch.tensor(img_high,dtype=torch.get_default_dtype())
       
        



class My_MedMNIST2D(MedMNIST2D):
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)
        
        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        

        
        img2 = load_decomposition_img(self.flag,self.split,index,self.labels[index])
        return img2, target
        



class RetinaMNIST(My_MedMNIST2D):
    flag = "retinamnist"

class BloodMNIST(My_MedMNIST2D):
    flag = "bloodmnist"


class OrganAMNIST(My_MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(My_MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(My_MedMNIST2D):
    flag = "organsmnist"
    
def get_data_class(typ):
    if typ=='retinamnist':
        return RetinaMNIST
    elif typ=='bloodmnist':
        return BloodMNIST
    elif typ=='organamnist':
        return OrganAMNIST
    elif typ=='organcmnist':
        return OrganCMNIST
    elif typ=='organsmnist':
        return OrganSMNIST



def get_train_val_test_data_2d(config):
    DataClass = get_data_class(config['dataset'])
    
    # img tranform
    if config['in_channel']==1:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
        val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
        
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=True, size=config['img_size'])
        val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=True, size=config['img_size'])
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=True, size=config['img_size'])
    
    
    return train_dataset,val_dataset,test_dataset
    
    

    







    
    
    
    