"""
Train CMC with AlexNet
"""
import os
import torch
import lightning as pl
import argparse
import wandb

from torchvision import transforms
from models.resnet import MyResNetsCMC, LightningContrastiveNet
from dataset import TouchFolderLabel



def parse_option():

    parser = argparse.ArgumentParser('Argument for testing')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=5, help='num of workers to use')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['resnet50t1', 'resnet101t1', 'resnet18t1',
                                                                        'resnet50t2', 'resnet101t2', 'resnet18t2',
                                                                        'resnet50t3', 'resnet101t3', 'resnet18t3'])
    parser.add_argument('--layer', type=int, default='5', help='layer to extract features from')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model')

    # dataset
    parser.add_argument('--dataset', type=str, default='touch_and_go', choices=['touch_and_go', 'object_folder'])
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')

    # add new views
    parser.add_argument('--view', type=str, default='Touch', choices=['Touch'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    opt = parser.parse_args()

    if not os.path.isdir(opt.data_folder) or not os.path.exists(opt.ckpt_path):
        raise ValueError(f'Data or model path not exist: {opt.data_folder} {opt.ckpt_path}')

    return opt


def get_test_loader(args):
    """Get the test loader"""

    data_folder = args.data_folder

    if args.view == 'Touch':
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset == 'touch_and_go':
        test_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain')
    elif args.dataset == 'object_folder':
        test_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain-of')
    else:
        raise NotImplementedError('data loader not supported {}'.format(args.data_loader))

    # test loader
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

    print(f'number of samples: {len(test_dataset)}')

    return test_loader


def get_model(args):
    """Instatiate the model to be tested"""

    if args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def test():
    """
    train the model with multiple GPUs using Lightning
    """
    # parse the args
    args = parse_option()

    # train loader
    test_loader = get_test_loader(args)

    # model and loss function
    model = get_model(args)

    model = LightningContrastiveNet(model, args)
    trainer = pl.Trainer(accelerator="gpu", 
                        devices=[0,1],
                        strategy="ddp")
    
    # Test from the checkpoint
    print(f"Loading model from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Perform the test
    trainer.test(model, test_loader)
    
    # check for checkpoints to resume from
    # ckpt_path = ""
    # if len(os.listdir(args.model_folder)) > 0:
    #     max_epoch = 0
    #     max_epoch_file = ""
    #     for file in os.listdir(args.model_folder):
    #         if file.endswith(".ckpt"):
    #             epoch = int(file.split("_")[-1])
    #             if epoch > max_epoch:
    #                 max_epoch = epoch
    #                 max_epoch_file = file
    #     ckpt_path = os.path.join(args.model_folder, max_epoch_file)
    
    # if os.path.exists(ckpt_path):
    #     trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    #     print(f"Resuming from checkpoint {ckpt_path}")
    # else:
    #     trainer.fit(model, train_loader)
        


if __name__ == '__main__':
    test()
