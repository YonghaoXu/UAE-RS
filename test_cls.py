import os
import numpy as np
import argparse
from tools.utils import *
from torch import nn
from dataset.scene_dataset import *
from torch.utils import data
import tools.model as models  

def main(args):
    if args.dataID==1:
        DataName = 'UCM'
        num_classes = 21
        classname = ('agricultural','airplane','baseballdiamond',
                        'beach','buildings','chaparral',
                        'denseresidential','forest','freeway',
                        'golfcourse','harbor','intersection',
                        'mediumresidential','mobilehomepark','overpass',
                        'parkinglot','river','runway',
                        'sparseresidential','storagetanks','tenniscourt')     

    elif args.dataID==2:        
        DataName = 'AID'
        num_classes = 30
        classname = ('airport','bareland','baseballfield',
                        'beach','bridge','center',
                        'church','commercial','denseresidential',
                        'desert','farmland','forest',
                        'industrial','meadow','mediumresidential',
                        'mountain','parking','park',
                        'playground','pond','port',
                        'railwaystation','resort','river',
                        'school','sparseresidential','square',
                        'stadium','storagetanks','viaduct')               

    adv_root_dir = args.save_path_prefix+DataName+'_adv/'+args.attack_func+'/'+args.surrogate_model+'/'
    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size,args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    adv_loader = data.DataLoader(
        scene_dataset(root_dir=adv_root_dir,pathfile='./dataset/'+DataName+'_test.txt', transform=composed_transforms, mode='adv'),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    clean_loader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset/'+DataName+'_test.txt', transform=composed_transforms),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
   
    ###################Target target_model Definition###################
    if args.target_model=='alexnet':
        target_model = models.alexnet(pretrained=False)
        target_model.classifier._modules['6'] = nn.Linear(4096, num_classes)    
    elif args.target_model=='vgg16':
        target_model = models.vgg16(pretrained=False)  
        target_model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif args.target_model=='vgg19':
        target_model = models.vgg19(pretrained=False)  
        target_model.classifier._modules['6'] = nn.Linear(4096, num_classes)        
    elif args.target_model=='resnet18':
        target_model = models.resnet18(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)        
    elif args.target_model=='resnet50':
        target_model = models.resnet50(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)        
    elif args.target_model=='resnet101':
        target_model = models.resnet101(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)        
    elif args.target_model=='resnext50_32x4d':
        target_model = models.resnext50_32x4d(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)        
    elif args.target_model=='resnext101_32x8d':
        target_model = models.resnext101_32x8d(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)
    elif args.target_model=='densenet121':
        target_model = models.densenet121(pretrained=False)
        target_model.classifier = nn.Linear(1024, num_classes)
    elif args.target_model=='densenet169':
        target_model = models.densenet169(pretrained=False)  
        target_model.classifier = nn.Linear(1664, num_classes)        
    elif args.target_model=='densenet201':
        target_model = models.densenet201(pretrained=False)  
        target_model.classifier = nn.Linear(1920, num_classes)        
    elif args.target_model=='inception':
        target_model = models.inception_v3(pretrained=True, aux_logits=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)
    elif args.target_model=='regnet_x_400mf':
        target_model = models.regnet_x_400mf(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)
    elif args.target_model=='regnet_x_8gf':
        target_model = models.regnet_x_8gf(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)
    elif args.target_model=='regnet_x_16gf':
        target_model = models.regnet_x_16gf(pretrained=False)  
        target_model.fc = torch.nn.Linear(target_model.fc.in_features, num_classes)
         
    dirpath = args.save_path_prefix+DataName+'/Pretrain/'+args.target_model+'/'
        
    model_path = os.listdir(dirpath)
    for filename in model_path: 
        filepath = os.path.join(dirpath, filename)           
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(dirpath, filename))
            model_path_resume = os.path.join(dirpath, filename)
        
    saved_state_dict = torch.load(model_path_resume)
    
    new_params = target_model.state_dict().copy()
    
    for i,j in zip(saved_state_dict,new_params):    
        new_params[j] = saved_state_dict[i]

    target_model.load_state_dict(new_params)

    target_model = torch.nn.DataParallel(target_model).cuda()
    target_model.eval()
    
    OA_clean,_ = test_acc(target_model,classname, clean_loader, 1,num_classes,print_per_batches=10)
    OA_adv,_ = test_acc(target_model,classname, adv_loader, 1,num_classes,print_per_batches=10)
    print('Clean Test Set OA:',OA_clean*100)
    print(args.attack_func+' Test Set OA:',OA_adv*100)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/',help='dataset path.')   
    parser.add_argument('--surrogate_model', type=str, default='resnet18',help='alexnet,resnet18,densenet121,regnet_x_400mf')      
    parser.add_argument('--target_model', type=str, default='inception',
                        help='alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf')
    parser.add_argument('--save_path_prefix', type=str,default='./')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--attack_func', type=str, default='fgsm',help='fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut')
    
    main(parser.parse_args())
