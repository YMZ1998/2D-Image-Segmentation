import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

torch.manual_seed(3407)

from utils.preprocess import pre_process

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class MyDataset(Dataset):
    def __init__(self, path, image_size=224):
        # self.images = torch.tensor(np.load(image_path))  # 加载npy数据
        # self.masks = torch.tensor(np.load(mask_path))
        self.path = path
        self.image_size = image_size
        self.image_paths = sorted(os.listdir(path + '/image'))
        mask_names = set(os.listdir(path + '/mask'))
        missing_masks = [name for name in self.image_paths if name not in mask_names]
        extra_masks = sorted(mask_names - set(self.image_paths))
        if missing_masks or extra_masks:
            raise ValueError(f"Image/mask mismatch in {path}: missing={missing_masks}, extra={extra_masks}")
        self.mask_paths = self.image_paths
        # self.transforms = transform
        self.images = []
        self.masks = []
        # self.load_all_images()

    def load_all_images(self):
        print('loading all images...')
        for i in tqdm(range(len(self.image_paths)), file=sys.stdout):
            image_path = os.path.join(self.path, 'image', self.image_paths[i])
            mask_path = os.path.join(self.path, 'mask', self.mask_paths[i])
            image, mask = pre_process(image_path, mask_path, self.image_size)
            self.images.append(image)
            self.masks.append(mask)
        print('loading all images finished!')
    def __getitem__(self, index):
        image_path = os.path.join(self.path, 'image', self.image_paths[index])
        mask_path = os.path.join(self.path, 'mask', self.mask_paths[index])
        image, mask = pre_process(image_path, mask_path, self.image_size)

        # image, mask = self.images[index], self.masks[index]

        # if self.transforms is not None:
        #     image = self.transforms(image)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        # mask = torch.Tensor(mask)
        # mask = self.transforms(mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)

def plot_data_loader_image(data_loader):
    plot_num = data_loader.batch_size

    for image, target in data_loader:
        plot_images = []
        plot_masks = []
        for i in range(plot_num):
            img = image[i].numpy().transpose(1, 2, 0)
            img = (img + 1) * 127.5
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.uint8(img)

            plot_masks.append(target[i])
            plot_images.append(img)

        fig, axes = plt.subplots(plot_num // 4, 8, figsize=(10, 10))
        axes = axes.flatten()
        j = 0
        for img, lab in zip(plot_images, plot_masks):
            axes[j].imshow(img)
            axes[j].axis("off")
            axes[j + 1].imshow(lab)
            axes[j + 1].axis("off")
            j += 2
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    dataset = MyDataset(args.data_path + "/augmentation_test", args.image_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, j in data_loader:
        print(i.shape, j.shape)
        print(np.min(i.numpy()), np.max(i.numpy()))
        print(np.min(j.numpy()), np.max(j.numpy()))
        break
    # dataset = MyDataset(os.path.join(args.data_path, 'augmentation_test'), args.image_size)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    plot_data_loader_image(data_loader)
