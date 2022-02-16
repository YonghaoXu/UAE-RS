import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.seg_dataset import seg_dataset
from model.Networks import *
import torchvision.models as models

epsilon = 1e-14

def main(args):
    if args.dataID==1:
        DataName = 'Vaihingen'
        num_classes = 5
        name_classes = np.array(['impervious surfaces','buildings','low vegetation','trees','cars'], dtype=np.str)
        train_list = './dataset/vaihingen_train.txt'
        test_list = './dataset/vaihingen_test.txt'
        root_dir = args.root_dir+DataName
    elif args.dataID==2:        
        DataName = 'Zurich'
        num_classes = 8
        name_classes = np.array(['Roads','Buildings','Trees','Grass','Bare Soil','Water','Rails','Pools'], dtype=np.str)
        train_list = './dataset/zurich_train.txt'
        test_list = './dataset/zurich_test.txt'
        root_dir = args.root_dir+DataName

    snapshot_dir = args.snapshot_dir+DataName+'/Pretrain/'+args.model+'/'

    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size_test.split(','))
    input_size_test = (w, h)
    w, h = map(int, args.input_size_train.split(','))
    input_size_train = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network
    if args.model=='fcn32s':
        model = fcn32s(n_classes=num_classes)
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i,j in zip(saved_state_dict,new_params):
            if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
                new_params[j] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.model=='fcn16s':
        model = fcn16s(n_classes=num_classes)
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i,j in zip(saved_state_dict,new_params):
            if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
                new_params[j] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.model=='fcn8s':
        model = fcn8s(n_classes=num_classes)
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i,j in zip(saved_state_dict,new_params):
            if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
                new_params[j] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.model=='deeplabv2':
        model = deeplab(num_classes=num_classes)
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i,j in zip(saved_state_dict,new_params):
            if (i[0] !='f')&(i[0] != 's')&(i[0] != 'u'):
                new_params[j] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.model=='deeplabv3_plus': 
        model = deeplabv3_plus(n_classes=num_classes)        
    elif args.model=='segnet':
        model = segnet(n_classes=num_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)   
    elif args.model=='icnet': 
        model = icnet(n_classes=num_classes)
    elif args.model=='contextnet': 
        model = contextnet(n_classes=num_classes)
    elif args.model=='sqnet': 
        model = sqnet(n_classes=num_classes)
    elif args.model=='pspnet':
        model = pspnet(n_classes=num_classes)        
    elif args.model=='unet': 
        model = unet(n_classes=num_classes)
    elif args.model=='linknet':
        model = linknet(n_classes=num_classes)
    elif args.model=='frrna': 
        model = frrn(n_classes=num_classes,model_type='A')
    elif args.model=='frrnb': 
        model = frrn(n_classes=num_classes,model_type='B')

    model.train()
    model = model.cuda()

    composed_transforms = transforms.Compose([         
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    src_loader = data.DataLoader(
                    seg_dataset(root_dir, train_list, transform=composed_transforms, max_iters=args.num_steps_stop*args.batch_size,
                    crop_size=input_size_train, mode='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(
                    seg_dataset(root_dir, test_list, transform=composed_transforms, mode='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # interpolation for the probability maps and labels 
    interp_train = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='bilinear')
    interp_test = nn.Upsample(size=(input_size_test[1], input_size_test[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,3)) 
    seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    for batch_index, src_data in enumerate(src_loader):
        if batch_index==args.num_steps_stop:
            break
        tem_time = time.time()
        model.train()
        optimizer.zero_grad()        
        adjust_learning_rate(optimizer,args.learning_rate,batch_index,args.num_steps)
        images, labels, _, _ = src_data
        images = images.cuda()      
        _,pre = model(images)   
        
        pre_output = interp_train(pre)
              
        # CE Loss
        labels = labels.cuda().long()
        seg_loss_value = seg_loss(pre_output, labels)
        _, predict_labels = torch.max(pre_output, 1)
        predict_labels = predict_labels.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        batch_oa = np.sum(predict_labels==labels)*1./len(labels.reshape(-1))
            
        hist[batch_index,0] = seg_loss_value.item()
        hist[batch_index,1] = batch_oa
        
        seg_loss_value.backward()
        optimizer.step()

        hist[batch_index,-1] = time.time() - tem_time
        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d Time: %.2f Batch OA = %.1f seg_loss = %.3f'%(batch_index+1,args.num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
                  
    model.eval()
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((num_classes, 1))

    for index, batch in enumerate(test_loader):  
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
                    _,pred = model(image_part)
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
    
        
        print(index+1, '/', len(test_loader), ': Testing ', name)

        # evaluate one image
        TP,FP,TN,FN,n_valid_sample = eval_image(test_pred.reshape(-1),label.reshape(-1),num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
    

    for i in range(num_classes):
        print('===>' + name_classes[i] + ': %.2f'%(F1[i] * 100))
    mF1 = np.mean(F1)    
    print('===> mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))    
    print('Save Model')                     
    model_name = 'batch'+repr(batch_index+1)+'_mF1_'+repr(int(mF1*10000))+'.pth'
    torch.save(model.state_dict(), os.path.join(
        snapshot_dir, model_name))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/',help='dataset path.')    
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='the index of the label ignored in the training.')
    parser.add_argument('--input_size_train', type=str, default='256,256',
                        help='width and height of input training images.')         
    parser.add_argument('--input_size_test', type=str, default='256,256',
                        help='width and height of input test images.')       
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of images in each batch.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for multithread dataloading.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='base learning rate.')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of training steps.')
    parser.add_argument('--num_steps_stop', type=int, default=5000,
                        help='Number of training steps for early stopping.')
    parser.add_argument('--restore_from', type=str, default='/iarai/home/yonghao.xu/PreTrainedModel/fcn8s_from_caffe.pth',
                        help='pretrained vgg16 model.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='regularisation parameter for L2-loss.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum component of the optimiser.')
    parser.add_argument('--model', type=str, default='fcn8s',
                        help='fcn8s,fcn16s,fcn32s,deeplabv2,deeplabv3_plus,segnet,icnet,contextnet,sqnet,pspnet,unet,linknet,frrna,frrnb')
    parser.add_argument('--snapshot_dir', type=str, default='./',
                        help='where to save snapshots of the model.')

    main(parser.parse_args())
