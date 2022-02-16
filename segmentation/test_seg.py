import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from model.Networks import *
from dataset.seg_dataset import seg_dataset
import os
from utils.tools import *
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

epsilon = 1e-14

def main(args):
    if args.dataID==1:
        DataName = 'Vaihingen'
        num_classes = 5
        name_classes = np.array(['impervious surfaces','buildings','low vegetation','trees','cars'], dtype=np.str)
        test_list = './dataset/vaihingen_test.txt'
        root_dir = args.root_dir+DataName

    elif args.dataID==2:        
        DataName = 'Zurich'
        num_classes = 8
        name_classes = np.array(['Roads','Buildings','Trees','Grass','Bare Soil','Water','Rails','Pools'], dtype=np.str)
        test_list = './dataset/zurich_test.txt'
        root_dir = args.root_dir+DataName
    
    
    adv_dir = args.snapshot_dir+DataName+'_adv/'+args.attack_func+'/'+args.surrogate_model+'/'

    snapshot_dir = args.snapshot_dir+'Map/'+DataName+'/'+args.target_model+'/'
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    
    composed_transforms = transforms.Compose([         
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    adv_loader = data.DataLoader(
                    seg_dataset(root_dir, test_list, transform=composed_transforms,mode='adv',adv_dir=adv_dir),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    clean_loader = data.DataLoader(
                    seg_dataset(root_dir, test_list, transform=composed_transforms,mode='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp_test = nn.Upsample(size=(input_size_test[1], input_size_test[0]), mode='bilinear')

    if args.target_model=='fcn32s':
        target_model = fcn32s(n_classes=num_classes)    
    elif args.target_model=='fcn16s':
        target_model = fcn16s(n_classes=num_classes)  
    elif args.target_model=='fcn8s':
        target_model = fcn8s(n_classes=num_classes)    
    elif args.target_model=='deeplabv2':
        target_model = deeplab(num_classes=num_classes)  
    elif args.target_model=='deeplabv3_plus': 
        target_model = deeplabv3_plus(n_classes=num_classes)
    elif args.target_model=='segnet':
        target_model = segnet(n_classes=num_classes)
    elif args.target_model=='icnet': 
        target_model = icnet(n_classes=num_classes)
    elif args.target_model=='contextnet': 
        target_model = contextnet(n_classes=num_classes)
    elif args.target_model=='sqnet': 
        target_model = sqnet(n_classes=num_classes)
    elif args.target_model=='pspnet':
        target_model = pspnet(n_classes=num_classes)
    elif args.target_model=='unet': 
        target_model = unet(n_classes=num_classes)
    elif args.target_model=='linknet':
        target_model = linknet(n_classes=num_classes)
    elif args.target_model=='frrna': 
        target_model = frrn(n_classes=num_classes,model_type='A')
    elif args.target_model=='frrnb': 
        target_model = frrn(n_classes=num_classes,model_type='B')

    dirpath = './'+DataName+'/Pretrain/'+args.target_model+'/'
    model_path = os.listdir(dirpath)
    for filename in model_path: 
        filepath = os.path.join(dirpath, filename)           
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(dirpath, filename))
            model_path_resume = os.path.join(dirpath, filename)

    saved_state_dict = torch.load(model_path_resume)
    target_model.load_state_dict(saved_state_dict)

    target_model.eval()
    target_model.cuda()
    
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((num_classes, 1))
   
    for index, batch in enumerate(clean_loader):  
        image, label,_, name = batch
        label = label.squeeze().numpy()
        img_size = image.shape[2:] 

        block_size = input_size_test
        min_overlap = 100

        # crop the test images into patches
        y_end,x_end = np.subtract(img_size, block_size)
        x = np.linspace(0, x_end, int(np.ceil(x_end/np.float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
        y = np.linspace(0, y_end, int(np.ceil(y_end/np.float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

        test_pred = np.zeros(img_size)
            
        for j in range(len(x)):    
            for k in range(len(y)):            
                r_start,c_start = (y[k],x[j])
                r_end,c_end = (r_start+block_size[0],c_start+block_size[1])
                image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()
            
                with torch.no_grad():
                    _,pred = target_model(image_part)

                _,pred = torch.max(interp_test(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                
                
                if (j==0)and(k==0):
                    test_pred[r_start:r_end, c_start:c_end] = pred
                elif (j==0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                elif (j!=0)and(k==0):
                    test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                elif (j!=0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]
    
        print(index+1, '/', len(clean_loader), ': Testing ', name)
        
        TP,FP,TN,FN,n_valid_sample = eval_image(test_pred.reshape(-1),label.reshape(-1),num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample
      
        test_pred = np.asarray(test_pred, dtype=np.uint8)

        if args.dataID==1:
            output_col = index2bgr_v(test_pred)
            plt.imsave('%s/%s_%s.png' % (snapshot_dir, 'clean_', name[0].split('.')[0]),output_col)
            
        elif args.dataID==2:
            output_col = index2bgr_z(test_pred)
            plt.imsave('%s/%s_%s.png' % (snapshot_dir, 'clean_', name[0].split('.')[0]),output_col)
                    
    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)

    for i in range(num_classes):
        print('===>' + name_classes[i] + ': %.2f'%(F1[i] * 100))
    mF1 = np.mean(F1)    
    print('===> clean mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))

    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((num_classes, 1))
   
    for index, batch in enumerate(adv_loader):  
        image, label,_, name = batch
        label = label.squeeze().numpy()
        img_size = image.shape[2:] 

        block_size = input_size_test
        min_overlap = 100

        # crop the test images 
        y_end,x_end = np.subtract(img_size, block_size)
        x = np.linspace(0, x_end, int(np.ceil(x_end/np.float(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
        y = np.linspace(0, y_end, int(np.ceil(y_end/np.float(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

        test_pred = np.zeros(img_size)
            
        for j in range(len(x)):    
            for k in range(len(y)):            
                r_start,c_start = (y[k],x[j])
                r_end,c_end = (r_start+block_size[0],c_start+block_size[1])
                image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()
            
                with torch.no_grad():
                    _,pred = target_model(image_part)

                _,pred = torch.max(interp_test(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                
                
                if (j==0)and(k==0):
                    test_pred[r_start:r_end, c_start:c_end] = pred
                elif (j==0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                elif (j!=0)and(k==0):
                    test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                elif (j!=0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]
    
        print(index+1, '/', len(adv_loader), ': Testing ', name)
        
        TP,FP,TN,FN,n_valid_sample = eval_image(test_pred.reshape(-1),label.reshape(-1),num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample
      
        test_pred = np.asarray(test_pred, dtype=np.uint8)

        if args.dataID==1:
            output_col = index2bgr_v(test_pred)
            plt.imsave('%s/%s_%s.png' % (snapshot_dir, 'adv_', name[0].split('.')[0]),output_col)            
        elif args.dataID==2:
            output_col = index2bgr_z(test_pred)
            plt.imsave('%s/%s_%s.png' % (snapshot_dir, 'adv_', name[0].split('.')[0]),output_col)
                    
    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)

    for i in range(num_classes):
        print('===>' + name_classes[i] + ': %.2f'%(F1[i] * 100))
    mF1 = np.mean(F1)    
    print('===> '+args.attack_func+ ' mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/',help='dataset path.')   
    parser.add_argument('--input_size_test', type=str, default='512,512',help='width and height of input test images.')   
    parser.add_argument('--num_workers', type=int, default=1,help='number of workers for multithread dataloading.')             
    parser.add_argument('--snapshot_dir', type=str, default='./',help='path to save result.')
    parser.add_argument('--surrogate_model', type=str, default='fcn8s',help='fcn8s,unet,pspnet,linknet')
    parser.add_argument('--target_model', type=str, default='segnet',
                        help='fcn8s,fcn16s,fcn32s,deeplabv2,deeplabv3_plus,segnet,icnet,contextnet,sqnet,pspnet,unet,linknet,frrna,frrnb')
    parser.add_argument('--attack_func', type=str, default='mixup',help='fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut')
    
    main(parser.parse_args())