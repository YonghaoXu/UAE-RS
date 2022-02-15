from torch.utils.data import Dataset
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')
 
class scene_dataset(Dataset):
    def __init__(self, root_dir, pathfile, transform=None, loader=default_loader, mode='clean'):
        pf = open(pathfile, 'r')
        imgs = []
        if mode=='clean':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split()
                name = words[0].split('/')[-1].split('.')[0]
                imgs.append((root_dir+words[0],int(words[1]),name))
        elif mode=='adv':
            for line in pf:
                line = line.rstrip('\n')
                words = line.split()
                name = words[0].split('/')[-1].split('.')[0]
                imgs.append((root_dir+words[0].split('/')[-1].split('.')[0]+'_adv.png',int(words[1]),name))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        pf.close()
 
    def __getitem__(self, index):
        fn, label, name = self.imgs[index]
        img = self.loader(fn)        
        if self.transform is not None:            
            img = self.transform(img)       
        return img,label,name
 
    def __len__(self):
        return len(self.imgs)
 