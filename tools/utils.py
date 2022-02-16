import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms


def test_acc(model,classname, data_loader, epoch,num_classes, print_per_batches=10):

    model.eval()
    
    class_name_list = classname
    num_classes = len(classname)
    num_batches = len(data_loader)

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    total = 0
    correct = 0
    class_acc = np.zeros((num_classes,1))
    for batch_idx, data in enumerate(data_loader):

        images, labels = data[0].cuda(),data[1].cuda()
        batch_size = labels.size(0)
        _,outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total += batch_size
        correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
        if (batch_idx+1)%print_per_batches == 0:
            print('Epoch[%d]-Validation-[%d/%d] Batch OA: %.2f %%' % (epoch,batch_idx+1,num_batches,100.0 * (predicted == labels).sum().item() / batch_size))

    for i in range(num_classes):
        class_acc[i] = 1.0*class_correct[i] / class_total[i]
        print('---------------Accuracy of %12s : %.2f %%---------------' % (
            class_name_list[i], 100 * class_acc[i])) 
    acc = 1.0*correct / total
    print('---------------Epoch[%d]Validation-OA: %.2f %%---------------' % (epoch,100.0 * acc))
    print('---------------Epoch[%d]Validation-AA: %.2f %%---------------' % (epoch,100.0 * np.mean(class_acc)))
    return acc,class_acc


def preprocess_image(img,args):
    """
        Processes image for input
    """
    
    composed_transforms = transforms.Compose([
        transforms.Resize(size=(args.crop_size,args.crop_size)),            
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
