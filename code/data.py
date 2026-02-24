import numpy as np
import torch
from PIL import Image
from medmnist.dataset import MedMNIST2D



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
        
        return img, target


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
    
def get_2d_data_class(typ):
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
    elif typ=='chestmnist':
        return ChestMNIST




def get_2d_data(data_typ,size):
    DataClass = get_2d_data_class(data_typ)
    
    train_dataset = DataClass(split='train', download=True, size=size)
    val_dataset = DataClass(split='val', download=True, size=size)
    test_dataset = DataClass(split='test', download=True, size=size)
    
    train = [d for d in train_dataset]
    val   = [d for d in val_dataset]
    test  = [d for d in test_dataset]
    
    return train+test+val
    
    
    














    
    
    
    