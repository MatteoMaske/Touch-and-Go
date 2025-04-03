import torch
import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, low_dim=128, in_channel=3, width=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.base * 8 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer == 1:
            return x
        x = self.layer1(x)
        if layer == 2:
            return x
        x = self.layer2(x)
        if layer == 3:
            return x
        x = self.layer3(x)
        if layer == 4:
            return x
        x = self.layer4(x)
        if layer == 5:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if layer == 6:
            return x
        x = self.fc(x)
        x = self.l2norm(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class InsResNet50(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1):
        super(InsResNet50, self).__init__()
        self.encoder = resnet50(width=width)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)


class ResNetV1(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV1, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=0.5)
            self.ab_to_l = resnet50(in_channel=2, width=0.5)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=0.5)
            self.ab_to_l = resnet18(in_channel=2, width=0.5)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=0.5)
            self.ab_to_l = resnet101(in_channel=2, width=0.5)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV2(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV2, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=1)
            self.ab_to_l = resnet50(in_channel=2, width=1)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=1)
            self.ab_to_l = resnet18(in_channel=2, width=1)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=1)
            self.ab_to_l = resnet101(in_channel=2, width=1)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class ResNetV3(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetV3, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=1, width=2)
            self.ab_to_l = resnet50(in_channel=2, width=2)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=1, width=2)
            self.ab_to_l = resnet18(in_channel=2, width=2)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=1, width=2)
            self.ab_to_l = resnet101(in_channel=2, width=2)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab

class ResNetT1(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetT1, self).__init__()
        if name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=3, width=0.5)
            self.ab_to_l = resnet50(in_channel=3, width=0.5)
        elif name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=3, width=0.5)
            self.ab_to_l = resnet18(in_channel=3, width=0.5)
        elif name == 'resnet101':
            self.l_to_ab = resnet101(in_channel=3, width=0.5)
            self.ab_to_l = resnet101(in_channel=3, width=0.5)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        image = x[:,:3,:,:]
        touch = x[:,3:,:,:]
        
        feat_image = self.image_to_touch(image, layer)
        feat_touch = self.touch_to_image(touch, layer)
        return feat_image, feat_touch

class ResNetT2(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetT2, self).__init__()
        if name == 'resnet50':
            self.image_to_touch = resnet50(in_channel=3, width=1.0)
            self.touch_to_image = resnet50(in_channel=3, width=1.0)
        elif name == 'resnet18':
            self.image_to_touch = resnet18(in_channel=3, width=1.0)
            self.touch_to_image = resnet18(in_channel=3, width=1.0)
        elif name == 'resnet101':
            self.image_to_touch = resnet101(in_channel=3, width=1.0)
            self.touch_to_image = resnet101(in_channel=3, width=1.0)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        image = x[:,:3,:,:]
        touch = x[:,3:,:,:]

        feat_image = self.image_to_touch(image, layer)
        feat_touch = self.touch_to_image(touch, layer)
        return feat_image, feat_touch

class ResNetT3(nn.Module):
    def __init__(self, name='resnet50'):
        super(ResNetT3, self).__init__()
        if name == 'resnet50':
            self.image_to_touch = resnet50(in_channel=3, width=2.0)
            self.touch_to_image = resnet50(in_channel=3, width=2.0)
        elif name == 'resnet18':
            self.image_to_touch = resnet18(in_channel=3, width=2.0)
            self.touch_to_image = resnet18(in_channel=3, width=2.0)
        elif name == 'resnet101':
            self.image_to_touch = resnet101(in_channel=3, width=2.0)
            self.touch_to_image = resnet101(in_channel=3, width=2.0)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name))

    def forward(self, x, layer=7):
        image = x[:,:3,:,:]
        touch = x[:,3:,:,:]

        feat_image = self.image_to_touch(image, layer)
        feat_touch = self.touch_to_image(touch, layer)
        return feat_image, feat_touch
    
class MyResNetsCMC(nn.Module):
    def __init__(self, name='resnet50v1'):
        super(MyResNetsCMC, self).__init__()
        if name.endswith('t1'):
            self.encoder = ResNetT1(name[:-2])
        elif name.endswith('t2'):
            self.encoder = ResNetT2(name[:-2])
        elif name.endswith('t3'):
            self.encoder = ResNetT3(name[:-2])
        else:
            raise NotImplementedError('model not support: {}'.format(name))

        # self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)
    
class LightningContrastiveNet(L.LightningModule):
    def __init__(self, model, args, contrast=None, criterion_l=None, criterion_ab=None):
        super().__init__()
        self.model = model
        self.contrast = contrast
        self.criterion_l = criterion_l
        self.criterion_ab = criterion_ab
        self.args = args

    def forward(self, x, layer=7):
        feat_l, feat_ab = self.model(x, layer=layer)
        return feat_l, feat_ab

    def training_step(self, batch, _):
        inputs, _, index = batch
        inputs = inputs.float()

        feat_l, feat_ab = self.forward(inputs)
        out_l, out_ab = self.contrast(feat_l, feat_ab, index)

        # Compute losses
        l_loss = self.criterion_l(out_l)
        ab_loss = self.criterion_ab(out_ab)

        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        loss = l_loss + ab_loss

        # Log losses
        metrics = {"train_loss": loss, "l_loss": l_loss, "l_prob": l_prob, "ab_loss": ab_loss, "ab_prob": ab_prob}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return loss
    
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, _):
        inputs, labels, _ = batch
        inputs = inputs.float()

        # Only perform a forward pass; do not compute any losses.
        feat_l, feat_ab = self.forward(inputs, self.args.layer)
        self.test_outputs.append({"feat_l": feat_l, "feat_ab": feat_ab, "labels": labels})

        return {"feat_l": feat_l, "feat_ab": feat_ab, "labels": labels}
    
    #TODO: Check this functions and add TSNE to project on a 2D plane
    def on_test_epoch_end(self):
        # Aggregate all features and labels
        all_feat_l = torch.cat([output["feat_l"] for output in self.test_outputs], dim=0) # (test_set_size, 512, 7, 7)
        all_labels = torch.cat([output["labels"] for output in self.test_outputs], dim=0)

        # Global Average Pooling over the spatial dimensions (7, 7)
        features_pooled = F.adaptive_avg_pool2d(all_feat_l, (1, 1))  # now shape [test_set_size, 512, 1, 1]
        features_pooled = features_pooled.view(features_pooled.size(0), features_pooled.size(1))  # reshape to [test_set_size, 512]

        # Use TSNE to project the pooled features into a 2D space
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_pooled.cpu().detach().numpy())

        # Convert to numpy for plotting
        labels_np = all_labels.cpu().detach().numpy()

        # Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, ticks=range(7), label='Material ID')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Feature Visualization with Material Labels')
        plt.savefig('plots/feature_visualization.png')
        plt.show()


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                lr=self.args.learning_rate,
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        milestones = np.asarray(self.args.lr_decay_epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=self.args.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

