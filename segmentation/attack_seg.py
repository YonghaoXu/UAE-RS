import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from model.Networks import *
from dataset.seg_dataset import seg_dataset
from PIL import Image
from utils.tools import *
from tqdm import tqdm


def main(args):
    if args.dataID==1:
        DataName = 'Vaihingen'
        num_classes = 5
        test_list = './dataset/vaihingen_test.txt'
        root_dir = args.root_dir+DataName        
        if args.attack_func[:2] == 'mi':
            mix_file = './dataset/Vaihingen_'+args.attack_func+'_sample.png'
    elif args.dataID==2:        
        DataName = 'Zurich'
        num_classes = 8
        test_list = './dataset/zurich_test.txt'
        root_dir = args.root_dir+DataName
        if args.attack_func[:2] == 'mi':
            mix_file = './dataset/Zurich_'+args.attack_func+'_sample.png'

    snapshot_dir = args.snapshot_dir+DataName+'_adv/'+args.attack_func+'/'+args.surrogate_model+'/'

    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    cudnn.enabled = True
    cudnn.benchmark = True

    ###################Surrogate Model Definition###################
    if args.surrogate_model=='fcn8s':
        surrogate_model = fcn8s(n_classes=num_classes)  
    elif args.surrogate_model=='unet': 
        surrogate_model = unet(n_classes=num_classes)
    elif args.surrogate_model=='pspnet':
        surrogate_model = pspnet(n_classes=num_classes)
    elif args.surrogate_model=='linknet': 
        surrogate_model = linknet(n_classes=num_classes)
   
    dirpath = args.snapshot_dir+DataName+'/Pretrain/'+args.surrogate_model+'/'
    model_path = os.listdir(dirpath)
    for filename in model_path: 
        filepath = os.path.join(dirpath, filename)           
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(dirpath, filename))
            model_path_resume = os.path.join(dirpath, filename)

    saved_state_dict = torch.load(model_path_resume)
    surrogate_model.load_state_dict(saved_state_dict)

    surrogate_model.eval()
    surrogate_model.cuda()
   
    composed_transforms = transforms.Compose([         
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    test_loader = data.DataLoader(
                    seg_dataset(root_dir, test_list, transform=composed_transforms, mode='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    tbar = tqdm(test_loader)
    num_batches = len(test_loader)
    cls_loss = nn.CrossEntropyLoss(ignore_index=255)
    kl_loss = torch.nn.KLDivLoss()
    mse_loss = torch.nn.MSELoss(reduction='none')
    tpgd_loss = torch.nn.KLDivLoss(reduction='sum')
    
    alpha = 1
    num_iter = 5

    if args.attack_func == 'fgsm':
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')                   
            
            adv_im = interp_img(adv_im)
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            adv_im.requires_grad = True  
            
            _,pred = surrogate_model(adv_im)            
            pred_loss = cls_loss(interp_img(pred), label)

            grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

            adv_im = interp_adv(adv_im.detach() + args.epsilon*grad / torch.norm(grad,float('inf')))
            delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
            adv_im = (X + delta).detach()
            
            recreated_image = recreate_image(adv_im.cpu())  

            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')   

    elif args.attack_func == 'ifgsm':
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')
                   
            adv_im = interp_img(adv_im)
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            adv_im.requires_grad = True  
            
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,pred = surrogate_model(adv_im)
                pred_loss = cls_loss(interp_img(pred), label)
                grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                if i==(num_iter-1):                      
                    adv_im = interp_adv(adv_im.detach() + alpha*grad / torch.norm(grad,float('inf')))
                    delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                    adv_im = (X + delta).detach()
                else: 
                    adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                    delta = torch.clamp(adv_im - interp_img(X), min=-args.epsilon, max=args.epsilon)
                    adv_im = (interp_img(X) + delta).detach()                
                
                recreated_image = recreate_image(adv_im.cpu())                
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')   

    elif args.attack_func == 'cw':
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')
                   
            adv_im = interp_img(adv_im)
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            
            Y[Y==255] = 0
            label_mask = F.one_hot(interp_lab(Y.unsqueeze(0)).squeeze(0).long().cuda(), num_classes=num_classes) 
            label_mask = torch.moveaxis(label_mask, -1, 1)

            adv_im.requires_grad = True          

            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,pred = surrogate_model(adv_im)
                pred = interp_img(pred)
                correct_logit = torch.mean(torch.sum(label_mask * pred,dim=1)[0])     
                wrong_logit = torch.mean(torch.max((1 - label_mask) * pred, dim=1)[0])
                loss = -(correct_logit - wrong_logit + args.C)      
                grad = torch.autograd.grad(loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                if i==(num_iter-1):                      
                    adv_im = interp_adv(adv_im.detach() + alpha*grad / torch.norm(grad,float('inf')))
                    delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                    adv_im = (X + delta).detach()
                else: 
                    adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                    delta = torch.clamp(adv_im - interp_img(X), min=-args.epsilon, max=args.epsilon)
                    adv_im = (interp_img(X) + delta).detach()
                                
                recreated_image = recreate_image(adv_im.cpu())                
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')   

    elif args.attack_func == 'tpgd':
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')
                   
            adv_im = interp_img(adv_im)
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            adv_im.requires_grad = True  
            
            with torch.no_grad():
                _,logit_ori = surrogate_model(X)
                logit_ori = interp_img(logit_ori)
            
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,pred = surrogate_model(adv_im)
                
                pred_loss = tpgd_loss(F.log_softmax(interp_img(pred), dim=1),
                            F.softmax(logit_ori, dim=1))
                grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                if i==(num_iter-1):                      
                    adv_im = interp_adv(adv_im.detach() + alpha*grad / torch.norm(grad,float('inf')))
                    delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                    adv_im = (X + delta).detach()
                else: 
                    adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                    delta = torch.clamp(adv_im - interp_img(X), min=-args.epsilon, max=args.epsilon)
                    adv_im = (interp_img(X) + delta).detach()                
                
                recreated_image = recreate_image(adv_im.cpu())                
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)

            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')   

    elif args.attack_func == 'jitter':
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            Y[Y==255] = 0
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')
                   
            adv_im = interp_img(adv_im)
            X_interp = interp_img(X)
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            
            label_mask = F.one_hot(interp_lab(Y.unsqueeze(0)).squeeze(0).long().cuda(), num_classes=num_classes).float()   
            label_mask = torch.moveaxis(label_mask, -1, 1)

            adv_im.requires_grad = True          

            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,out = surrogate_model(adv_im)
                out = interp_img(out)                

                _, pre = torch.max(out, dim=1)
                wrong = (pre != label)
         
                norm_z = torch.norm(out, p=float('inf'), dim=1, keepdim=True)
                hat_z = nn.Softmax(dim=1)(args.scale*out/norm_z)
                hat_z = hat_z + args.std*torch.randn_like(hat_z)
                loss = mse_loss(hat_z, label_mask).mean(dim=1)

                norm_r = torch.norm((adv_im - X_interp), p=float('inf'), dim=[1])        
                nonzero_r = (norm_r != 0)
                loss[wrong*nonzero_r] /= norm_r[wrong*nonzero_r]

                loss = loss.mean()   
                grad = torch.autograd.grad(loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                if i==(num_iter-1):                      
                    adv_im = interp_adv(adv_im.detach() + alpha*grad / torch.norm(grad,float('inf')))
                    delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                    adv_im = (X + delta).detach()
                else: 
                    adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                    delta = torch.clamp(adv_im - interp_img(X), min=-args.epsilon, max=args.epsilon)
                    adv_im = (interp_img(X) + delta).detach()
                                
                recreated_image = recreate_image(adv_im.cpu())                
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')    

    elif args.attack_func == 'mixup' or args.attack_func == 'mixcut':
        mixup_im = composed_transforms(Image.open(mix_file).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            mixup_feature,_ = surrogate_model(mixup_im)
        for index, batch in enumerate(tbar):  
            X, Y,_, name = batch
            image_size = X.shape[2:] 
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            
            interp_img = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='bilinear')
            interp_lab = nn.Upsample(size=(int(image_size[0]*args.downsample), int(image_size[1]*args.downsample)), mode='nearest')
            interp_adv = nn.Upsample(size=(image_size[0], image_size[1]), mode='bilinear')
                    
            adv_im = interp_img(adv_im)
            momentum = torch.zeros_like(adv_im).cuda()
            label = interp_lab(label.unsqueeze(0)).squeeze(0).long()
            adv_im.requires_grad = True  

            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(index+1,num_batches,i+1))
                adv_im.requires_grad = True     

                feature,pred = surrogate_model(adv_im)
                feature_size = feature.shape[2:]             
                interp_feature = nn.Upsample(size=(feature_size[0], feature_size[1]), mode='bilinear')
                
                pred_loss = cls_loss(interp_img(pred), label)
                inception_loss = -kl_loss(feature, interp_feature(mixup_feature))           
            
                total = pred_loss*args.beta+inception_loss
            
                grad = torch.autograd.grad(total, adv_im,
                                        retain_graph=False, create_graph=False)[0]
                grad = grad / torch.norm(grad,p=1)
                grad = grad + momentum*args.decay
                momentum = grad

                if i==(num_iter-1):                      
                    adv_im = interp_adv(adv_im.detach() + alpha*grad / torch.norm(grad,float('inf')))
                    delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                    adv_im = (X + delta).detach()
                else: 
                    adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                    delta = torch.clamp(adv_im - interp_img(X), min=-args.epsilon, max=args.epsilon)
                    adv_im = (interp_img(X) + delta).detach()            
            
                recreated_image = recreate_image(adv_im.cpu())
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
                
            gen_name = name[0].split('.')[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(snapshot_dir+gen_name,'png')   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=1)    
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/',help='dataset path.')   
    parser.add_argument('--num_workers', type=int, default=0,help='number of workers for multithread dataloading.')           
    parser.add_argument('--snapshot_dir', type=str, default='./',help='where to save snapshots of the surrogate_model.')
    parser.add_argument('--attack_func', type=str, default='mixup',help='fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut')
    parser.add_argument('--surrogate_model', type=str, default='fcn8s',help='fcn8s,unet,pspnet,linknet')
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--C', type=float, default=50)
    parser.add_argument('--downsample', type=float, default=0.9)

    main(parser.parse_args())