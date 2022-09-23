from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import sys
sys.path.append("/data/CoANet-Road")
from dataloaders import custom_transforms as tr

class Segmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 1

    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = "/data/chusai_release/road"
        self._image_dir = "/data/chusai_release/train/images"
        self._cat_dir = os.path.join(self._base_dir, 'gt')#category
        self._con_dir = os.path.join(self._base_dir, 'connect_8_d1')
        self._con_d1_dir = os.path.join(self._base_dir, 'connect_8_d3')
        self.args = args
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        _splits_dir = os.path.join(self._base_dir)
        total=10778
        k=5
        train_n=int(total*(k-1)/k)
        if split=='train':
            self.im_ids=list(range(train_n))
        elif split=='val':
            self.im_ids=list(range(train_n,total))
        self.images=[os.path.join(self._image_dir, str(n)+'.tif') for n in self.im_ids]
        self.categories=[os.path.join(self._cat_dir, str(n)+'.png') for n in self.im_ids]
        self.connect_label=[os.path.join(self._con_dir, str(n)) for n in self.im_ids]
        self.connect_d1_label=[os.path.join(self._con_d1_dir, str(n)) for n in self.im_ids]

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size


    def __getitem__(self, index):
        _img, _target, _connect0, _connect1, _connect2, _connect_d1_0, _connect_d1_1, _connect_d1_2 = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'connect0': _connect0, 'connect1': _connect1, 'connect2': _connect2,
                  'connect_d1_0': _connect_d1_0, 'connect_d1_1': _connect_d1_1, 'connect_d1_2': _connect_d1_2}
        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _connect0 = Image.open(self.connect_label[index] + '_0.png').convert('RGB')
        _connect1 = Image.open(self.connect_label[index] + '_1.png').convert('RGB')
        _connect2 = Image.open(self.connect_label[index] + '_2.png').convert('RGB')
        _connect_d1_0 = Image.open(self.connect_d1_label[index] + '_0.png').convert('RGB')
        _connect_d1_1 = Image.open(self.connect_d1_label[index] + '_1.png').convert('RGB')
        _connect_d1_2 = Image.open(self.connect_d1_label[index] + '_2.png').convert('RGB')

        return _img, _target, _connect0, _connect1, _connect2, _connect_d1_0, _connect_d1_1, _connect_d1_2

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(45),#180
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])

        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512
    args.batch_size=16
    args.dataset = 'spacenet'

    voc_train = Segmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=False, num_workers=0)

    for ii, sample in enumerate(dataloader):
        if ii == 1:
            break
        for jj in range(sample["image"].size()[0]):
            if jj==3:
                break
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            # connect_d1_0 = sample['connect_d1_0'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='spacenet')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        

    # plt.show(block=True)
    plt.savefig("/data/test.png")


