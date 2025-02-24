import numpy as np
import torch
import torchvision.datasets as datasets
import os
import random
import cv2

from skimage import color
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import Optional

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, index


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img


class TouchFolderLabel(Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False, mode='train', label='full', data_amount=100):
        self.two_crop = two_crop
        self.dataroot = '/media/mmlab/Volume/matteomascherin/touch_and_go/dataset/' if "of" not in mode else "/media/mmlab/Volume/matteomascherin/material-classification/dataset/"
        self.mode = mode
        if mode == 'train':
            with open(os.path.join(root, 'train.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'train-of':
            with open(os.path.join(root, 'train_offull.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'test':
            with open(os.path.join(root, 'test_new_full.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'test-of':
            with open(os.path.join(root, 'test_offull.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'pretrain':
            with open(os.path.join(root, 'pretrain_OF.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'train-of-balanced':
            with open(os.path.join(root, 'train_offull_balanced.txt'),'r') as f:
                data = f.read().split('\n')
        elif mode == 'test-of-balanced':
            with open(os.path.join(root, 'test_offull_balanced.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            print('Mode other than train and test')
            exit()
        
        if mode == 'train' and label == 'rough':
            with open(os.path.join(root, 'train_rough.txt'),'r') as f:
                data = f.read().split('\n')
        
        if mode == 'test' and label == 'rough':
            with open(os.path.join(root, 'test_rough.txt'),'r') as f:
                data = f.read().split('\n')

        
        self.length = len(data)
        self.env = data
        self.transform = transform
        self.target_transform = target_transform
        self.label = label



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        
        assert index < self.length,'index_A range error'

        try:
            raw, target = self.env[index].strip().split(',')# mother path for A
        except ValueError:
            print(f"Trying to split '{self.env[index]}'")
            raise ValueError('Error in reading the dataset, __getitem__')
        
        target = int(target)
        if self.label == 'hard':
            if target == 7 or target == 8 or target == 9 or target == 11 or target == 13:
                target = 1
            else:
                target = 0
        
        # idx = os.path.basename(raw)
        # dir = self.dataroot + raw[:16]
        dir, idx = raw.split('/')
        dir = os.path.join(self.dataroot, dir)

        # load image and gelsight
        A_img_path = os.path.join(dir, 'video_frame', idx)
        A_gelsight_path = os.path.join(dir, 'gelsight_frame', idx)
        
        try:
            A_img = Image.open(A_img_path).convert('RGB')
            A_gel = Image.open(A_gelsight_path).convert('RGB')
        except:
            print(f"Error in opening image {A_img_path} or {A_gelsight_path}")
            raise ValueError('Error in opening the dataset, __getitem__')
        

        if self.transform is not None:
            A_img = self.transform(A_img)
            A_gel = self.transform(A_gel)

        out = torch.cat((A_img, A_gel), dim=0)
        
        if self.mode == 'pretrain':
            return out, target, index

        return out, target
    
    def __len__(self):
        """Return the total number of images."""
        return self.length
    
def load_image(image_path) -> Optional[np.ndarray]:
    """Try to load an image with PIL and OpenCV to check for corruption.
    Returns:
    - np.ndarray if the image is fine.
    - False if the image is corrupt.
    """
    try:
        # PIL check
        with Image.open(image_path) as img:
            img.verify()  # Verify without loading
        # OpenCV check
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError("Image could not be decoded")
        return img_cv2  # Image is fine
    except Exception as e:
        print(f"Corrupt image detected: {image_path} - {e}")
        return None

def show_batches(dataloader: DataLoader, n_batches=3):
    """Show batches of images using matplotlib subplots"""
    import matplotlib.pyplot as plt

    dataset_name = dataloader.dataset.mode
    os.makedirs(f'./batches/{dataset_name}', exist_ok=True)
    folder = f'./batches/{dataset_name}'

    for batch_id, batch in enumerate(dataloader):
        images, labels = batch
        touch_images = images[:, 3:]

        fig, axs = plt.subplots(4, 8, figsize=(16, 8))
        for i, ax in enumerate(axs.flat):
            ax.imshow(touch_images[i].permute(1, 2, 0).numpy())
            ax.axis('off')
        plt.savefig(os.path.join(folder, f'batch_{batch_id}.png'))

        if batch_id == n_batches - 1:
            break

def check_dataset_integrity(dataset: TouchFolderLabel):
    """Check images integrity trying to load all the images from the given dataset"""

    loop = tqdm(range(len(dataset)))
    # Scan dataset
    corrupt_images = []
    for idx in loop:
        try:
            out, target = dataset[idx]  # Load sample
            
            # Extract image paths
            raw = dataset.env[idx].strip().split(',')[0]
            dir, idx_name = raw.split('/')
            dir = os.path.join(dataset.dataroot, dir)

            A_img_path = os.path.join(dir, 'video_frame', idx_name)
            A_gelsight_path = os.path.join(dir, 'gelsight_frame', idx_name)

            # Check if images are corrupt
            rgb_image = load_image(A_img_path)
            touch_image = load_image(A_gelsight_path)

            if rgb_image is None or touch_image is None:
                corrupt_images.append((A_img_path, A_gelsight_path))
            else:
                touch_images.append(touch_image)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")

        show_batch(touch_images)

    # Print summary
    if corrupt_images:
        print("\n⚠️ Corrupt images found:")
        for img_pair in corrupt_images:
            print(img_pair)
    else:
        print("✅ No corrupt images detected!")

if __name__ == "__main__":
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]

    np.random.seed(0)
    torch.manual_seed(0)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = TouchFolderLabel(root='./dataset', mode='train-of-balanced', transform=train_transform, label='full')
    # check_dataset_integrity(dataset)

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    # show_batches(dataloader, n_batches=10)

    dataset = TouchFolderLabel(root='./dataset', mode='test', transform=train_transform, label='full')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    show_batches(dataloader, n_batches=10)
