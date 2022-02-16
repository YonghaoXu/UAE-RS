import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class Classifier_Module_VGG(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module_VGG, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out

class deeplab(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(deeplab, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
       
        self.classifier = Classifier_Module_VGG(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        f = self.features(x)        
        x = self.classifier(f)   
        return f,x

    def optim_parameters(self, args):
        return self.parameters()



# FCN32s
class fcn32s(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample(score, x.size()[2:])

        return conv1, out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn16s(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        out = F.upsample(score, x.size()[2:])

        return conv2,out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print(idx, l1, l2)
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


# FCN 8s
class fcn8s(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=True):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore4 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore8 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 16, stride=8, bias=False
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
            ]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[
                :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
            ]

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
            ]
            return conv2,out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])

        return conv2,out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class segnet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return down1,up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x1, logits

class conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation=1, is_batchnorm=True
    ):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        self.cb4 = conv2DBatchNorm(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(
            in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                is_batchnorm=is_batchnorm,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=1,
                bias=bias,
                dilation=1,
                is_batchnorm=is_batchnorm,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)

class residualBlockPSP(nn.Module):
    def __init__(
        self,
        n_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation=1,
        include_range="all",
        is_batchnorm=True,
    ):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ["all", "conv"]:
            layers.append(
                bottleNeckPSP(
                    in_channels,
                    mid_channels,
                    out_channels,
                    stride,
                    dilation,
                    is_batchnorm=is_batchnorm,
                )
            )
        if include_range in ["all", "identity"]:
            for i in range(n_blocks - 1):
                layers.append(
                    bottleNeckIdentifyPSP(
                        out_channels, mid_channels, stride, dilation, is_batchnorm=is_batchnorm
                    )
                )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class pyramidPooling(nn.Module):
    def __init__(
        self, in_channels, pool_sizes, model_name="pspnet", fusion_mode="cat", is_batchnorm=True
    ):
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=bias,
                    is_batchnorm=is_batchnorm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.training or self.model_name != "icnet":  # general settings or pspnet
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
        else:  # eval mode and icnet: pre-trained for 1025 x 2049
            #k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            #strides = [(5, 10), (10, 20), (16, 32), (33, 65)]
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))

        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                pp_sum = pp_sum + out

            return pp_sum

class pspnet(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105
    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow
    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928
    """

    def __init__(
        self, n_classes=21, block_config=[3, 4, 23, 3], input_size=(473, 473), version=None
    ):

        super(pspnet, self).__init__()

        self.block_config = block_config
        self.n_classes = n_classes
        self.input_size = input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)


    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        f = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(f, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)


        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.interpolate(x, size=inp_shape, mode="bilinear", align_corners=True)

        return f,x

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out+residual)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True))

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)

        # solution for padding issues
        # diffY = x_low_level.size()[2] - x_high_level.size()[2]
        # diffX = x_low_level.size()[3] - x_high_level.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])

        x = self.conv2(x)

        return x

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w -max_width) // 2
    return layer[:,:,diffy:(diffy + max_height),diffx:(diffx + max_width)]


def up_pad(layer, skip_height, skip_width):
    _, _, h, w = layer.size()
    diffy = skip_height - h
    diffx = skip_width -w
    return F.pad(layer,[diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])


class LinkNetImprove(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()

        base = models.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)


    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y


class linknet(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, n_classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)


        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)


        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(f)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = f + self.decoder1(d2, f)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)


        return f,y

def get_interp_size(input, s_factor=1, z_factor=1):  # for caffe
    ori_h, ori_w = input.shape[2:]
    # shrink (s_factor >= 1)
    ori_h = (ori_h) / s_factor
    ori_w = (ori_w) / s_factor

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h) * (z_factor - 1)
    ori_w = ori_w + (ori_w) * (z_factor - 1)

    resize_shape = (int(ori_h), int(ori_w))
    return resize_shape



class cascadeFeatureFusion(nn.Module):
    def __init__(
        self, n_classes, low_in_channels, high_in_channels, out_channels, is_batchnorm=True
    ):
        super(cascadeFeatureFusion, self).__init__()

        bias = not is_batchnorm

        self.low_dilated_conv_bn = conv2DBatchNorm(
            low_in_channels,
            out_channels,
            3,
            stride=1,
            padding=2,
            bias=bias,
            dilation=2,
            is_batchnorm=is_batchnorm,
        )
        self.low_classifier_conv = nn.Conv2d(
            int(low_in_channels),
            int(n_classes),
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            dilation=1,
        )  # Train only
        self.high_proj_conv_bn = conv2DBatchNorm(
            high_in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

    def forward(self, x_low, x_high):
        x_low_upsampled = F.interpolate(
            x_low, size=get_interp_size(x_low, z_factor=2), mode="bilinear", align_corners=True
        )

        low_cls = self.low_classifier_conv(x_low_upsampled)

        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        
        interp = nn.Upsample(size=(high_fm.shape[2], high_fm.shape[3]), mode='bilinear')
        high_fused_fm = F.relu(interp(low_fm) + high_fm, inplace=True)

        return high_fused_fm, low_cls





class icnet(nn.Module):

    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545
    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow
    """

    def __init__(
        self,
        n_classes=19,
        block_config=[3, 4, 6, 3],
        input_size=(1025, 2049),
        version=None,
        is_batchnorm=True,
    ):

        super(icnet, self).__init__()

        bias = not is_batchnorm

        self.block_config = block_config 
        self.n_classes = n_classes
        self.input_size = input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(
            self.block_config[0], 64, 32, 128, 1, 1, is_batchnorm=is_batchnorm
        )
        self.res_block3_conv = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="conv",
            is_batchnorm=is_batchnorm,
        )
        self.res_block3_identity = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="identity",
            is_batchnorm=is_batchnorm,
        )

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(
            self.block_config[2], 256, 128, 512, 1, 2, is_batchnorm=is_batchnorm
        )
        self.res_block5 = residualBlockPSP(
            self.block_config[3], 512, 256, 1024, 1, 4, is_batchnorm=is_batchnorm
        )

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(
            1024, [6, 3, 2, 1], model_name="icnet", fusion_mode="sum", is_batchnorm=is_batchnorm
        )

        # Final conv layer with kernel 1 in sub4 branch
        self.conv5_4_k1 = conv2DBatchNormRelu(
            in_channels=1024,
            k_size=1,
            n_filters=256,
            padding=0,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu2_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu3_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.classification = nn.Conv2d(128, self.n_classes, 1, 1, 0)

        # Cascade Feature Fusion Units
        self.cff_sub24 = cascadeFeatureFusion(
            self.n_classes, 256, 256, 128, is_batchnorm=is_batchnorm
        )
        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 128, 64, 128, is_batchnorm=is_batchnorm
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # H, W -> H/2, W/2
        x_sub2 = F.interpolate(
            x, size=get_interp_size(x, s_factor=2), mode="bilinear", align_corners=True
        )

        # H/2, W/2 -> H/4, W/4
        x_sub2 = self.convbnrelu1_1(x_sub2)
        f = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(f)

        # H/4, W/4 -> H/8, W/8
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)

        # H/8, W/8 -> H/16, W/16
        x_sub2 = self.res_block2(x_sub2)
        x_sub2 = self.res_block3_conv(x_sub2)
        # H/16, W/16 -> H/32, W/32
        x_sub4 = F.interpolate(
            x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)

        x_sub4 = self.res_block4(x_sub4)
        x_sub4 = self.res_block5(x_sub4)

        x_sub4 = self.pyramid_pooling(x_sub4)
        x_sub4 = self.conv5_4_k1(x_sub4)

        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.convbnrelu3_sub1(x_sub1)

        x_sub24, sub4_cls = self.cff_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.cff_sub12(x_sub24, x_sub1)

        x_sub12 = F.interpolate(
            x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)
        sub124_cls = self.classification(x_sub12)

        sub124_cls = F.interpolate(
            sub124_cls,
            size=get_interp_size(sub124_cls, z_factor=4),
            mode="bilinear",
            align_corners=True,
        )
        return f,sub124_cls

class conv2DGroupNormRelu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs

class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False, n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv2DGroupNorm(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            self.conv1 = conv2DBatchNormRelu(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )
            self.conv2 = conv2DBatchNorm(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming

class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False, n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv_unit(
                out_channels,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32, out_channels, k_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = conv_unit(
                out_channels, out_channels, k_size=3, stride=1, padding=1, bias=False
            )

        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s * self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode="nearest")
        z_prime = z + x

        return y_prime, z_prime

frrn_specs_dic = {
    "A": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]],
        "decoder": [[2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
    "B": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]],
        "decoder": [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
}
class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=21, model_type="B", group_norm=False, n_groups=16):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(
                RU(
                    channels=48,
                    kernel_size=3,
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )
            self.down_residual_units.append(
                RU(
                    channels=48,
                    kernel_size=3,
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0, stride=1, bias=False)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels

        self.merge_conv = nn.Conv2d(
            prev_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False
        )

        self.classif_conv = nn.Conv2d(
            48, self.n_classes, kernel_size=1, padding=0, stride=1, bias=True
        )

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        f = y
        z = self.split_conv(x)

        prev_channels = 48
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels
        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode="bilinear", align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat(
            [F.upsample(y, scale_factor=2, mode="bilinear", align_corners=True), z], dim=1
        )
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return f,x



class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        # self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ELU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        # out1 = self.bn2(out1)
        out2 = self.conv3(x)
        # out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out


class sqnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.num_classes = n_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1) # 32
        # self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire1_1 = Fire(96, 16, 64)
        self.fire1_2 = Fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire2_1 = Fire(128, 32, 128)
        self.fire2_2 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire3_1 = Fire(256, 64, 256)
        self.fire3_2 = Fire(512, 64, 256)
        self.fire3_3 = Fire(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(96)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(192, self.num_classes, 3, stride=2, padding=1, output_padding=1)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1) # 32

        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x_1 = self.relu1(x)
        # print "x_1: %s" % str(x_1.size())
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        # print "x_2: %s" % str(x_2.size())
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        # print "x_3: %s" % str(x_3.size())
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        f = self.fire3_3(x)
        x = self.parallel(f)
        # print "x: %s" % str(x.size())
        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        # print "y_3: %s" % str(y_3.size())
        # x = x.transpose(1, 2, 0)
        # print('x_3.size():', x_3.size())
        # print('y_3.size():', y_3.size())
        x_3 = F.interpolate(x_3, y_3.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        # concat x_3
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        # print "y_2: %s" % str(y_2.size())
        # concat x_2
        # print('x_2.size():', x_2.size())
        # print('y_2.size():', y_2.size())
        y_2 = F.interpolate(y_2, x_2.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        # print "y_1: %s" % str(y_1.size())
        # concat x_1
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return f,x #, x_1, x_2, x_3, y_1, y_2, y_3




class Custom_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthSepConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            Custom_Conv(in_channels, in_channels * t, 1),
            DepthConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out



    
class Shallow_net(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128, **kwargs):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(3, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)


    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x

class Deep_net(nn.Module):
    def __init__(self, in_channels, block_channels,
                 t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks

        self.conv_ = Custom_Conv(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(LinearBottleneck, block_channels[2], block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(LinearBottleneck, block_channels[3], block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(LinearBottleneck, block_channels[4], block_channels[5], num_blocks[5], t[5], 1)

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h,w), mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x



class contextnet(nn.Module):
    def __init__(self, n_classes, aux=False, **kwargs):
        super(contextnet, self).__init__()
        self.aux = aux
        self.spatial_detail = Shallow_net(32, 64, 128)
        self.context_feature_extractor = Deep_net(32, [32, 32, 48, 64, 96, 128], [1, 6, 6, 6, 6, 6], [1, 1, 3, 3, 2, 2])
        self.feature_fusion = FeatureFusionModule(128, 128, 128)
        self.classifier = Classifer(128, n_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, n_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]

        higher_res_features = self.spatial_detail(x)

        x_low = F.interpolate(x, scale_factor = 0.25, mode='bilinear', align_corners=True)

        x = self.context_feature_extractor(x_low)

        x = self.feature_fusion(higher_res_features, x)

        x = self.classifier(x)

        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return higher_res_features,x



model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print("Having loaded imagenet-pretrained successfully!")

def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class deeplabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(deeplabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),

                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x5,x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k






