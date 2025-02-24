from __future__ import print_function
from torch import channel_shuffle
from torchmetrics.classification import Accuracy
import torchvision

import torch
import torch.nn as nn
import torchmetrics
import lightning as L
import torch.optim as optim
import numpy as np



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class LinearClassifierResNet(nn.Module):
    def __init__(self, layer=6, n_label=1000):
        super(LinearClassifierResNet, self).__init__()
        self.layer = layer
        if layer == 1:
            nChannels = 64
        elif layer == 2:
            nChannels = 64
        elif layer == 3:
            nChannels = 128
        elif layer == 4:
            nChannels = 256
        elif layer == 5:
            nChannels = 512
        elif layer == 6:
            nChannels = 512
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()
        self.classifier.add_module('LiniearClassifier', nn.Linear(nChannels, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.layer < 6:
            avg_pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
            x = avg_pool(x).squeeze()
        return self.classifier(x)
    
class LightningLinearProb(L.LightningModule):
    def __init__(self, model, classifier, criterion, args):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.criterion = criterion
        self.top1_acc = Accuracy(task="multiclass", num_classes=args.n_label, top_k=1)
        self.top3_acc = Accuracy(task="multiclass", num_classes=args.n_label, top_k=3)
        self.args = args

    def forward(self, input):
        with torch.no_grad():
            feat_image, feat_touch = self.model(input, self.args.layer)
            if self.args.test_modality == 'touch':
                feat = feat_touch.detach()
            else:
                feat = feat_image.detach()

        output = self.classifier(feat)
        return output
    
    def training_step(self, batch, batch_idx):

        input, target = batch
        input = input.float()

        output = self.forward(input)
        loss = self.criterion(output, target)

        acc1 = self.top1_acc(output, target)
        acc3 = self.top3_acc(output, target)

        metrics = {'train_loss': loss, 'top1_acc_train': acc1, 'top3_acc_train': acc3, 'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        input = input.float()

        output = self.forward(input)
        loss = self.criterion(output, target)

        acc1 = self.top1_acc(output, target)
        acc3 = self.top3_acc(output, target)

        print(output.argmax(dim=1))

        metrics = {'val_loss': loss, 'top1_acc_val': acc1, 'top3_acc_val': acc3}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        # SGD variant
        optimizer = optim.SGD(self.classifier.parameters(),
                            lr=self.args.learning_rate,
                            momentum=self.args.momentum,
                            weight_decay=self.args.weight_decay)
        
        milestones = np.asarray(self.args.lr_decay_epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=self.args.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # Adam variant
        # optimizer = optim.Adam(self.classifier.parameters(), lr=self.args.learning_rate)
        # return optimizer