import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tf


def random_crop(image,mask,crop_size=(512,512)):
    not_valid = True
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
        label = np.asarray(mask_crop, np.float32)
        if np.sum(label.reshape(-1)<255)>0:
            not_valid = False
    return image_crop,mask_crop 


class seg_dataset(data.Dataset):
    def __init__(self, root_dir, list_path, transform=None, max_iters=None,crop_size=(256, 256), mode='train',adv_dir=None):
        self.list_path = list_path
        self.crop_size = crop_size
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.transform = transform
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if mode == 'adv':
            for name in self.img_ids:
                img_file = osp.join(adv_dir, "%s" % name.split('.')[0]+'_adv.png')
                label_file = osp.join(root_dir, "gt/%s" % name.replace('tif','png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
        else:
            for name in self.img_ids:
                img_file = osp.join(root_dir, "img/%s" % name)
                label_file = osp.join(root_dir, "gt/%s" % name.replace('tif','png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
        
        
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
     
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        if self.mode=='train':
            image,label = random_crop(image,label,self.crop_size)
         
        if self.transform is not None:            
            image = self.transform(image)   
        size = image.shape
        label = np.asarray(label, np.float32)

        return image, label.copy(), np.array(size), name
