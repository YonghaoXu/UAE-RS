import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
 

def index2bgr_z(c_map):

    # mapping W x H x 1 class index to W x H x 3 BGR image
    im_col, im_row = np.shape(c_map)
    c_map_r = np.ones((im_col, im_row), 'uint8')*255
    c_map_g = np.ones((im_col, im_row), 'uint8')*255
    c_map_b = np.ones((im_col, im_row), 'uint8')*255
    c_map_r[c_map == 0] = 0
    c_map_r[c_map == 1] = 100
    c_map_r[c_map == 2] = 0
    c_map_r[c_map == 3] = 0
    c_map_r[c_map == 4] = 150
    c_map_r[c_map == 5] = 0
    c_map_r[c_map == 6] = 255
    c_map_r[c_map == 7] = 150
    c_map_g[c_map == 0] = 0
    c_map_g[c_map == 1] = 100
    c_map_g[c_map == 2] = 125
    c_map_g[c_map == 3] = 255
    c_map_g[c_map == 4] = 80
    c_map_g[c_map == 5] = 0
    c_map_g[c_map == 6] = 255
    c_map_g[c_map == 7] = 150
    c_map_b[c_map == 0] = 0
    c_map_b[c_map == 1] = 100
    c_map_b[c_map == 2] = 0
    c_map_b[c_map == 3] = 0
    c_map_b[c_map == 4] = 0
    c_map_b[c_map == 5] = 150
    c_map_b[c_map == 6] = 0
    c_map_b[c_map == 7] = 255
    c_map_rgb = np.zeros((im_col, im_row, 3), 'uint8')
    c_map_rgb[:, :, 0] = c_map_r
    c_map_rgb[:, :, 1] = c_map_g
    c_map_rgb[:, :, 2] = c_map_b
    
    return c_map_rgb

def index2bgr_v(c_map):

    # mapping W x H x 1 class index to W x H x 3 BGR image
    im_col, im_row = np.shape(c_map)
    c_map_r = np.ones((im_col, im_row), 'uint8')*255
    c_map_g = np.ones((im_col, im_row), 'uint8')*255
    c_map_b = np.ones((im_col, im_row), 'uint8')*255
    c_map_r[c_map == 0] = 255
    c_map_r[c_map == 1] = 0
    c_map_r[c_map == 2] = 0
    c_map_r[c_map == 3] = 0
    c_map_r[c_map == 4] = 255
    c_map_g[c_map == 0] = 255
    c_map_g[c_map == 1] = 0
    c_map_g[c_map == 2] = 255
    c_map_g[c_map == 3] = 255
    c_map_g[c_map == 4] = 255
    c_map_b[c_map == 0] = 255
    c_map_b[c_map == 1] = 255
    c_map_b[c_map == 2] = 255
    c_map_b[c_map == 3] = 0
    c_map_b[c_map == 4] = 0
    
    c_map_rgb = np.zeros((im_col, im_row, 3), 'uint8')
    c_map_rgb[:, :, 0] = c_map_r
    c_map_rgb[:, :, 1] = c_map_g
    c_map_rgb[:, :, 2] = c_map_b
    
    return c_map_rgb


def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)



def preprocess_image(img,args):
    """
        Processes image for input
    """
    
    composed_transforms = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    im_as_var = composed_transforms(img)
    im_as_var = Variable(im_as_var.unsqueeze(0)).cuda().requires_grad_()
    return im_as_var
    

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = im_as_var.data.numpy()[0].copy()
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    # recreated_im = recreated_im[..., ::-1]
    return recreated_im
    