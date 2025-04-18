"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import lightning as pl
import argparse
import wandb

from torchvision import transforms, datasets
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from util import adjust_learning_rate, AverageMeter

from models.resnet import MyResNetsCMC, LightningContrastiveNet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from NCE.sup_con_loss import SupConLoss

from dataset import TouchFolderLabel


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=[
                                                                        'resnet50t1', 'resnet101t1', 'resnet18t1',
                                                                        'resnet50t2', 'resnet101t2', 'resnet18t2',
                                                                        'resnet50t3', 'resnet101t3', 'resnet18t3'])
    # parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--supconloss', action='store_true', help='using Supervised Contrastive Loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='touch_and_go', choices=['touch_and_go', 'object_folder', 'object_folder_balanced'],)

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')

    # add new views
    parser.add_argument('--view', type=str, default='touch', choices=['touch', 'visual'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # data amount
    parser.add_argument('--comment', type=str, default='', help='comment')

    # wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb')
    parser.add_argument('--wandb_name', type=str, default=None, help='username of wandb')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path')


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'supcon' if opt.supconloss else 'nce'
    if opt.supconloss:
        opt.nce_k = 0

    if opt.dataset == 'object_folder_balanced':
        dataset = "ofb"
    elif opt.dataset == 'object_folder':
        dataset = "of"
    elif opt.dataset == 'touch_and_go':
        dataset = "tg"
    else:
        raise ValueError('dataset not supported {}'.format(opt.dataset))

    opt.model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}_dataset_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size, dataset)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder = args.data_folder

    if args.view == 'touch' or args.view == 'vision':
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    else:
        raise ValueError('view not implemented {}'.format(args.view))

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'touch_and_go':
        train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain')
    elif args.dataset == 'object_folder':
        train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain-of')
    elif args.dataset == 'object_folder_balanced':
        train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain-ofb')
    else:
        raise NotImplementedError('data loader not supported {}'.format(args.dataset))
    
    # train loader
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    if args.supconloss:
        # Supervised Contrastive Loss
        contrastive_criterion = SupConLoss()
    else:
        # NCE loss - from original paper
        contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, False)
        criterion_l = NCECriterion(n_data)
        criterion_ab = NCECriterion(n_data)
        contrastive_criterion = (contrast, criterion_l, criterion_ab)

    print(f"Current loss function: {contrastive_criterion}")

    if torch.cuda.is_available():
        model = model.cuda()
        if args.supconloss:
            contrastive_criterion = contrastive_criterion.cuda()
        else:
            contrastive_criterion[0] = contrastive_criterion[0].cuda()
            contrastive_criterion[1] = contrastive_criterion[1].cuda()
            contrastive_criterion[2] = contrastive_criterion[2].cuda()
        cudnn.benchmark = True

    return model, contrastive_criterion


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            # index = index.cuda(non_blocking=True)
            index = index.cuda()
            inputs = inputs.cuda()

        # ===================forward=====================
        feat_l, feat_ab = model(inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, index)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'image_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                'touch_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lprobs=l_prob_meter,
                abprobs=ab_prob_meter))
            # print(out_l.shape)
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # wandb
    if args.wandb == True:
        wandb.init(project="visuo-tactile-cmc", entity=args.wandb_name, name=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            'epochs': args.epochs,
            "lr_decay_epochs": args.lr_decay_epochs,
            "batch_size": args.batch_size, 
            "lr_decay_rate": args.lr_decay_rate,
            "comment": args.comment
            }
        wandb.watch(model)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        
        if args.wandb == True:
            wandb.log({"image_loss_epoch": l_loss, "image_prob_epoch": l_prob, "touch_loss_epoch": ab_loss, "touch_prob_epoch": ab_prob})

        torch.cuda.empty_cache()
    
    if args.wandb == True:
        wandb.finish()

def train_parallelized():
    """
    train the model with multiple GPUs using Lightning
    """
    # parse the args
    args = parse_option()

    # train loader
    train_loader, n_data = get_train_loader(args)

    # model and loss function
    model, contrastive_criterion = set_model(args, n_data)

    #Logger
    if args.wandb:
        wandb = WandbLogger(project="visuo-tactile-cmc", name=args.model_name)
        wandb.watch(model)

    checkpoint_name = "ckpt_{epoch}_{step}"
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_folder, 
                                        monitor="train_loss",
                                        mode="min", # "max" for accuracy, "min" for loss
                                        save_top_k=3, 
                                        filename=checkpoint_name)
    
    model = LightningContrastiveNet(model, args, contrastive_criterion)
    trainer = pl.Trainer(accelerator="gpu", 
                        devices=[0,1],
                        strategy="ddp",
                        max_epochs=args.epochs,
                        callbacks=[checkpoint_callback],
                        logger=wandb if args.wandb else None)
    
    if args.resume and os.path.exists(args.resume):
        ckpt_path = args.resume
        trainer.fit(model, train_loader, ckpt_path=ckpt_path)
        print(f"Resuming from checkpoint {ckpt_path}")
    else:
        trainer.fit(model, train_loader)


if __name__ == '__main__':
    # main()
    train_parallelized()
