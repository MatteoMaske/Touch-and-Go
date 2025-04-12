""" Test and plot the learned features from the model trained on a Touch-and-Go contrastive network. """
import os
import torch
import lightning as pl
import argparse

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
    parser.add_argument('--layer', type=int, default=5, help='layer to extract features from')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model')

    # dataset
    parser.add_argument('--dataset', type=str, default='touch_and_go', choices=['touch_and_go', 'object_folder', 'object_folder_balanced'])
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')

    # add new views
    parser.add_argument('--view', type=str, default='Touch', choices=['Touch'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    opt = parser.parse_args()

    if not os.path.isdir(opt.data_folder) or not os.path.exists(opt.ckpt_path):
        raise ValueError(f'Data or model path not exist: {opt.data_folder} {opt.ckpt_path}')
    
    # Name to be used for the plot
    backbone_dataset = opt.ckpt_path.split('/')[-1].split('_')[0]
    material_dataset = "of" if "object_folder" in opt.dataset else "tg"
    opt.exp_name = f'{backbone_dataset}_backbone_{material_dataset}_materials'

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
    elif args.dataset == 'object_folder_balanced':
        test_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='pretrain-ofb')
    else:
        raise NotImplementedError('data loader not supported {}'.format(args.data_loader))

    # test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

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
                        devices=[0],
                        strategy="auto")
    
    # Test from the checkpoint
    print(f"Loading model from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    lightning_ckpt = {}
    if ".ckpt" in args.ckpt_path:
        lightning_ckpt = checkpoint['state_dict']
        model.load_state_dict(lightning_ckpt, strict=False) # lightning format
    else:
        # Creating an articial checkpoint to add "model." to every key
        for k in checkpoint['model']:
            new_k = "model." + k
            new_k = new_k.replace("module.", "")
            lightning_ckpt[new_k] = checkpoint['model'][k]
        model.load_state_dict(lightning_ckpt)

    count = 0
    for k in lightning_ckpt:
        if k in model.state_dict():
            count += 1 
    print(f"Loaded {count} keys from the checkpoint")

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
