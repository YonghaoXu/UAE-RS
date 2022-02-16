import os
import argparse
from tools.utils import *
from torch.nn import functional as F
import tools.model as models  
from dataset.scene_dataset import *
from torch import nn
from torch.utils import data
from tqdm import tqdm

def main(args):
    if args.dataID==1:
        DataName = 'UCM'
        num_classes = 21
        if args.attack_func[:2] == 'mi':
            mix_file = './dataset/UCM_'+args.attack_func+'_sample.png'
    elif args.dataID==2:        
        DataName = 'AID'
        num_classes = 30
        if args.attack_func[:2] == 'mi':
            mix_file = './dataset/AID_'+args.attack_func+'_sample.png'
                    
    save_path_prefix = args.save_path_prefix+DataName+'_adv/'+args.attack_func+'/'+args.surrogate_model+'/'

    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)

    composed_transforms = transforms.Compose([
            transforms.Resize(size=(args.crop_size,args.crop_size)),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    imloader = data.DataLoader(
        scene_dataset(root_dir=args.root_dir,pathfile='./dataset/'+DataName+'_test.txt', transform=composed_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ###################Surrogate Model Definition###################
    if args.surrogate_model=='alexnet':
        surrogate_model = models.alexnet(pretrained=False)
        surrogate_model.classifier._modules['6'] = nn.Linear(4096, num_classes)            
    elif args.surrogate_model=='resnet18':
        surrogate_model = models.resnet18(pretrained=False)  
        surrogate_model.fc = torch.nn.Linear(surrogate_model.fc.in_features, num_classes)        
    elif args.surrogate_model=='densenet121':
        surrogate_model = models.densenet121(pretrained=False)
        surrogate_model.classifier = nn.Linear(1024, num_classes)
    elif args.surrogate_model=='regnet_x_400mf':
        surrogate_model = models.regnet_x_400mf(pretrained=False)  
        surrogate_model.fc = torch.nn.Linear(surrogate_model.fc.in_features, num_classes)
                
    surrogate_model_path = args.save_path_prefix+DataName+'/Pretrain/'+args.surrogate_model+'/'
    model_path = os.listdir(surrogate_model_path)
    for filename in model_path: 
        filepath = os.path.join(surrogate_model_path, filename)           
        if os.path.isfile(filepath) and filename.lower().endswith('.pth'):
            print(os.path.join(surrogate_model_path, filename))
            model_path_resume = os.path.join(surrogate_model_path, filename)

    saved_state_dict = torch.load(model_path_resume)
    new_params = surrogate_model.state_dict().copy()

    for i,j in zip(saved_state_dict,new_params):        
        new_params[j] = saved_state_dict[i]

    surrogate_model.load_state_dict(new_params)

    surrogate_model = torch.nn.DataParallel(surrogate_model).cuda()
    surrogate_model.eval()


    num_batches = len(imloader)
    
    kl_loss = torch.nn.KLDivLoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    tpgd_loss = torch.nn.KLDivLoss(reduction='sum')
    mse_loss = torch.nn.MSELoss(reduction='none')
    tbar = tqdm(imloader)
           
    num_iter = 5
    alpha = 1

    if args.attack_func == 'fgsm':
        for batch_index, src_data in enumerate(tbar):        
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()            

            tbar.set_description('Batch: %d/%d' %(batch_index+1,num_batches))
            adv_im.requires_grad = True     
            _,out = surrogate_model(adv_im)
            pred_loss = cls_loss(out, label)
            grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

            adv_im = adv_im.detach() + args.epsilon*grad / torch.norm(grad,float('inf'))
            delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
            adv_im = (X + delta).detach()
            
            recreated_image = recreate_image(adv_im.cpu())        
            
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')   
                
    elif args.attack_func == 'ifgsm':
        for batch_index, src_data in enumerate(tbar):       
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(batch_index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,out = surrogate_model(adv_im)
                pred_loss = cls_loss(out, label)
                grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()
                
                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')   

    elif args.attack_func == 'cw':
        for batch_index, src_data in enumerate(tbar):        
            X, Y, img_name = src_data
            X = X.cuda()
            label_mask = F.one_hot(Y, num_classes=num_classes).cuda()    
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()            

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(batch_index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,out = surrogate_model(adv_im)
                
                correct_logit = torch.sum(label_mask * out)     
                wrong_logit = torch.max((1 - label_mask) * out)
                loss = -(correct_logit - wrong_logit + args.C)       

                grad = torch.autograd.grad(loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]


                adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()
                
                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')   

    elif args.attack_func == 'tpgd':
        for batch_index, src_data in enumerate(tbar):    
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            Y = Y.numpy().squeeze()
            _,logit_ori = surrogate_model(X)
            logit_ori = logit_ori.detach()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(batch_index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,logit_adv = surrogate_model(adv_im)
                pred_loss = tpgd_loss(F.log_softmax(logit_adv, dim=1),
                            F.softmax(logit_ori, dim=1))
                
                grad = torch.autograd.grad(pred_loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]


                adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()
                
                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')   

    elif args.attack_func == 'jitter':
        for batch_index, src_data in enumerate(tbar):        
            X, Y, img_name = src_data
            X = X.cuda()
            label_mask = F.one_hot(Y, num_classes=num_classes).cuda().float()  
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(batch_index+1,num_batches,i+1))
                adv_im.requires_grad = True     
                _,out = surrogate_model(adv_im)
                
                _, pre = torch.max(out, dim=1)
                wrong = (pre != label)

                norm_z = torch.norm(out, p=float('inf'), dim=1, keepdim=True)
                hat_z = nn.Softmax(dim=1)(args.scale*out/norm_z)

                hat_z = hat_z + args.std*torch.randn_like(hat_z)

                loss = mse_loss(hat_z, label_mask).mean(dim=1)

                norm_r = torch.norm((adv_im - X), p=float('inf'), dim=[1,2,3])
                nonzero_r = (norm_r != 0)
                loss[wrong*nonzero_r] /= norm_r[wrong*nonzero_r]

                loss = loss.mean()

                grad = torch.autograd.grad(loss, adv_im,
                                        retain_graph=False, create_graph=False)[0]


                adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()
                
                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
            
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')  

    elif args.attack_func == 'mixup' or args.attack_func == 'mixcut':
        mixup_im = composed_transforms(Image.open(mix_file).convert('RGB')).unsqueeze(0).cuda()
        mixup_feature,_ = surrogate_model(mixup_im)
        mixup_feature = mixup_feature.data
        for batch_index, src_data in enumerate(tbar):        
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()
            momentum = torch.zeros_like(X).cuda()
            
            # Start iteration
            for i in range(num_iter):
                tbar.set_description('Batch: %d/%d, Iteration:%d' %(batch_index+1,num_batches,i+1))      
                adv_im.requires_grad = True     
                pred_loss = 0
                mix_loss = 0
                for k in range(5):
                    # Scale augmentation
                    feature,pred = surrogate_model(adv_im/(2**(k)))                
                    pred_loss += cls_loss(pred, label)
                    mix_loss += -kl_loss(feature, mixup_feature)
            
                total = pred_loss*args.beta+mix_loss           
                grad = torch.autograd.grad(total, adv_im,
                                        retain_graph=False, create_graph=False)[0]

                grad = grad / torch.norm(grad,p=1)
                grad = grad + momentum*args.decay
                momentum = grad

                adv_im = adv_im.detach() + alpha*grad / torch.norm(grad,float('inf'))
                delta = torch.clamp(adv_im - X, min=-args.epsilon, max=args.epsilon)
                adv_im = (X + delta).detach()
                
                recreated_image = recreate_image(adv_im.cpu())
                adv_im = preprocess_image(Image.fromarray(recreated_image),args)
                
            gen_name = img_name[0]+'_adv.png'
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix+gen_name,'png')   
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=1)    
    parser.add_argument('--surrogate_model', type=str, default='resnet18',help='alexnet,resnet18,densenet121,regnet_x_400mf')
    parser.add_argument('--save_path_prefix', type=str,default='./')
    parser.add_argument('--root_dir', type=str, default='/iarai/home/yonghao.xu/Data/',help='dataset path.')   
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--attack_func', type=str, default='mixup',help='fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut')
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--C', type=float, default=50)

    main(parser.parse_args())
