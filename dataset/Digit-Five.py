from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch


# load mnist data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_mnist(data_dir, size=28):
    """
    Args:
        data_dir: the path of Digit-Five dataset
        size=28: image size (28 or 32)
    Return:
        train_data: [55000, size, size, 3]
        test_data: [10000, size, size, 3]
        train_label: [55000,]
        test_label: [10000,]
    """
    if not size in [28, 32]:
        raise ValueError('Size for mnist data should be 28 or 32')

    # mnist data mat path
    file_path = os.path.join(data_dir, 'mnist_data.mat')
    data = loadmat(file_path)
    # type(data): dict
    # keys: 'test_32', 'test_28', 'label_test', 'label_train', 'train_32', 'train_28'

    # data['train_32']: [55000, 32, 32]
    # data['train_28']: [55000, 28, 28, 1]
    # data['label_train']: [55000, 10]

    # value range: [0, 255]
    train_data = np.reshape(data['train_' + str(size)], (55000, size, size, 1))
    # train_data: [55000, size, size, 1]
    train_data = np.repeat(train_data, 3, axis=3)
    # train_data: [55000, size, size, 3]
    test_data = np.reshape(data['test_' + str(size)], (10000, size, size, 1))
    # test_data: [10000, size, size, 1]
    test_data = np.repeat(test_data, 3, axis=3)
    # test_data: [10000, size, size, 3]

    # labels are 0, 1, 2, ..., 9
    train_label = np.nonzero(data['label_train'])[1]
    # train_label: [55000,]
    test_label = np.nonzero(data['label_test'])[1]
    # test_label: [10000,]

    return train_data, test_data, train_label, test_label


# load mnistm data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_mnistm(data_dir):
    """
    Args:
        data_dir: the path of Digit-Five dataset
    Return:
        train_data: [55000, 28, 28, 3]
        test_data: [10000, 28, 28, 3]
        train_label: [55000,]
        test_label: [10000,]
    """
    # mnistm data mat path
    file_path = os.path.join(data_dir, 'mnistm_with_label.mat')
    data = loadmat(file_path)
    # type(data): dict
    # keys: 'label_test': [10000, 10], 'label_train': [550000, 10], 'test': [10000, 28, 28, 3], 'train': [55000, 28, 28, 3]

    # value range: [0, 255]
    train_data = np.array(data['train'])
    # train_data: [55000, 28, 28, 3]
    test_data = np.array(data['test'])
    # test_data: [10000, 28, 28, 3]

    # labels are 0, 1, 2, ..., 9
    train_label = np.nonzero(data['label_train'])[1]
    # train_label: [55000,]
    test_label = np.nonzero(data['label_test'])[1]
    # test_label: [10000,]

    return train_data, test_data, train_label, test_label


# load svhn data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_svhn(data_dir):
    """
    Args:
        data_dir: the path of Digit-Five dataset
    Return:
        train_data: [73257, 32, 32, 3]
        test_data: [26032, 32, 32, 3]
        train_label: [73257,]
        test_label: [10000,]
    """
    # svhn train data mat path
    train_file_path = os.path.join(data_dir, 'svhn_train_32x32.mat')
    # svhn test data mat path
    test_file_path = os.path.join(data_dir, 'svhn_test_32x32.mat')
    # train data
    train_data = loadmat(train_file_path)
    # test data
    test_data = loadmat(test_file_path)
    # type(data): dict
    # keys: 'X': [32, 32, 3, N], 'y': [N, 1]

    # value range: [0, 255]
    train_data_arr = np.array(train_data['X']).transpose((3, 0, 1, 2))
    # train_data_arr: [73257, 32, 32, 3]
    test_data_arr = np.array(test_data['X']).transpose((3, 0, 1, 2))
    # test_data: [26032, 32, 32, 3]

    # NOTE: labels are 1, 2, 3, ..., 10, where 10 represents '0' actually!
    train_label = np.array(train_data['y'])[:, 0]
    # replace label 10 with 0, an alternative is to use np.where(condition, x, y)
    train_label[train_label == 10] = 0
    # train_label: [73257,]

    # NOTE: labels are 1, 2, 3, ..., 10, where 10 represents '0' actually!
    test_label = np.array(test_data['y'])[:, 0]
    # replace label 10 with 0, an alternative is to use np.where(condition, x, y)
    test_label[test_label == 10] = 0
    # test_label: [10000,]

    return train_data_arr, test_data_arr, train_label, test_label


# load syn data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_syn(data_dir):
    """
    Args:
        data_dir: the path of Digit-Five dataset
    Return:
        train_data: [25000, 32, 32, 3]
        test_data: [9000, 32, 32, 3]
        train_label: [25000,]
        test_label: [9000,]
    """
    # syn data mat path
    file_path = os.path.join(data_dir, 'syn_number.mat')
    data = loadmat(file_path)
    # type(data): dict
    # keys: 'train_data', 'train_label', 'test_data', 'test_label'

    # data['train_data']: [25000, 32, 32, 3]
    # data['train_label']: [25000, 1]
    # data['test_data']: [9000, 32, 32, 3]
    # data['test_label']: [9000, 1]

    # value range: [0, 255]
    train_data = np.array(data['train_data'])
    # train_data: [25000, 32, 32, 3]
    test_data = np.array(data['test_data'])
    # test_data: [9000, 32, 32, 3]

    # labels are 0, 1, 2, ..., 9
    train_label = np.array(data['train_label'])[:, 0]
    # train_label: [25000,]
    test_label = np.array(data['test_label'])[:, 0]
    # test_label: [9000,]

    return train_data, test_data, train_label, test_label


# load usps data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_usps(data_dir):
    """
    Args:
        data_dir: the path of Digit-Five dataset
    Return:
        train_data: [7438, 28, 28, 3]
        test_data: [1860, 28, 28, 3]
        train_label: [7438,]
        test_label: [1860,]
    """
    # usps data mat path
    file_path = os.path.join(data_dir, 'usps_28x28.mat')
    data = loadmat(file_path)
    # type(data): dict
    # key: 'dataset': [2, 2]

    # dataset[0][0]: [7438, 1, 28, 28]
    # dataset[0][1]: [7438, 1]
    # dataset[1][0]: [1860, 1, 28, 28]
    # dataset[1][1]: [1860, 1]

    # value range: [0, 1) -> [0, 255)
    train_data = np.array(data['dataset'][0][0] * 255).astype(np.uint8).transpose((0, 2, 3, 1))
    # train_data: [7438, 28, 28, 1]
    train_data = np.repeat(train_data, 3, axis=3)
    # train_data: [7438, 28, 28, 3]
    test_data = np.array(data['dataset'][1][0] * 255).astype(np.uint8).transpose((0, 2, 3, 1))
    # test_data: [1860, 28, 28, 1]
    test_data = np.repeat(test_data, 3, axis=3)
    # test_data: [1860, 28, 28, 3]

    # labels are 0, 1, 2, ..., 9
    train_label = np.array(data['dataset'][0][1])[:, 0]
    # train_label: [7438,]
    test_label = np.array(data['dataset'][1][1])[:, 0]
    # test_label: [1860,]

    return train_data, test_data, train_label, test_label


# implement __init__(), __len__(), and __getitem__() for customized datasets
# Digit-Five dataset for a single domain
class DigitFiveDatasetSingle(Dataset):
    """
    `Digit-Five` Dataset for a single domain

    Args:
        data_path (string): Source path of Digit-Five dataset
        domain (string): select a domain of Digit-Five dataset
        train (bool): Load train set or test set
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g., `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it
    """
    def __init__(self, data_path, domain, train=True, transform=None, target_transform=None):
        self.data_path = data_path
        self.domain = domain
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.domains = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']

        # domain check
        if not self.domain in self.domains:
            raise ValueError(self.domain + ' is an unknown domain for Digit-Five dataset')

        # function call str: load_xxx(self.data_path)
        func = 'load_' + self.domain + '(self.data_path)'
        # load data
        self.train_data, self.test_data, self.train_label, self.test_label = eval(func)

    def __len__(self):
        if self.train:
            return len(self.train_label)
        else:
            return len(self.test_label)

    def __getitem__(self, idx) -> dict:
        """
        Args:
            idx: index of the data sample. If idx >= len, randomly pick a sample instead.
        Return: a dict
            data['image']: PIL Image of shape (H, W, C) (if not transformed)
            data['label']: the corresponding label, int
        """
        # randomly pick one sample if idx is out of range
        if idx >= len(self):
            idx = np.random.randint(0, len(self))

        data = dict()
        if self.train:
            # load train set
            # image
            # Image.array takes a numpy array of shape (H, W, C)
            if self.transform is not None:
                data['image'] = self.transform(Image.fromarray(self.train_data[idx], mode='RGB'))
            else:
                data['image'] = Image.fromarray(self.train_data[idx], mode='RGB')
            # label
            if self.target_transform is not None:
                data['label'] = self.target_transform(self.train_label[idx])
            else:
                data['label'] = self.train_label[idx]
        else:
            # load test set
            # image
            if self.transform is not None:
                data['image'] = self.transform(Image.fromarray(self.test_data[idx], mode='RGB'))
            else:
                data['image'] = Image.fromarray(self.test_data[idx], mode='RGB')
            # label
            if self.target_transform is not None:
                data['label'] = self.target_transform(self.test_label[idx])
            else:
                data['label'] = self.test_label[idx]

        return data

    def display(self):
        """
        Display basic information of the dataset
        """
        print('domain:', self.domain)
        print('train:', self.train)
        print('train data:', self.train_data.shape)
        print('train label:', self.train_label.shape)
        print('test data:', self.test_data.shape)
        print('test label:', self.test_label.shape)


# DigitFiveDataset
# NOTE: domains are ALIGNED
class DigitFiveDataset(Dataset):
    """
    `Digit-Five` Dataset

    Args:
        data_path (string): Source path of Digit-Five dataset
        train (bool): Load train set or test set
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g., `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it
    """
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.domains = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']

        # five digit datasets, a dict
        self.dataset = dict()
        # self.dataset['xxx'] = DigitFiveDatasetSingle(self.data_path, domain='xxx', train=self.train)
        for domain in self.domains:
            self.dataset[domain] = eval('DigitFiveDatasetSingle(self.data_path, domain=\'' + domain + '\', train=self.train)')

    def __len__(self):
        length = 0
        # use max length
        for domain in self.domains:
            if len(self.dataset[domain]) > length:
                length = len(self.dataset[domain])
        return length

    def __getitem__(self, idx) -> dict:
        """
        Args:
            idx: index of the data sample (same idx for all domains)
        Return: a dict, data[DOMAIN] is also a dict
            data[DOMAIN]['image']: PIL Image of shape (H, W, C) (if not transformed)
            data[DOMAIN]['label']: the corresponding label, int
        """
        data = dict()
        for domain in self.domains:
            data[domain] = dict()

        # image
        if self.transform is not None:
            for domain in self.domains:
                data[domain]['image'] = self.transform(self.dataset[domain][idx]['image'])
        else:
            for domain in self.domains:
                data[domain]['image'] = self.dataset[domain][idx]['image']
        # label
        if self.target_transform is not None:
            for domain in self.domains:
                data[domain]['label'] = self.target_transform(self.dataset[domain][idx]['label'])
        else:
            for domain in self.domains:
                data[domain]['label'] = self.dataset[domain][idx]['label']

        return data

    def display(self):
        notice_str = '**************************************************'
        print(notice_str)
        for domain in self.domains:
            self.dataset[domain].display()
            print(notice_str)


# TODO: unaligned dataloader
class MultiDomainDataLoader(object):
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        # assume we have 3 batches of data
        self.num_batch = 3
        self.data = np.random.rand(self.num_batch * self.batch_size, 3, 28, 28)
        self.batch = 0

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == len(self):
            raise StopIteration()
        else:
            self.batch += 1
            return self.data[(self.batch - 1) * self.batch_size: self.batch * self.batch_size]


# display an image
def visualize_img(img):
    """
    Args:
        img: image Tensor of size [C, H, W]
    """
    # img: [C, H, W]
    # reverse normalize
    img = img * 0.5 + 0.5
    # reverse ToTensor scaling
    img *= 255
    # convert type to uint8
    img = img.type(torch.uint8)
    # Tensor to PIL
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    plt.imshow(img)
    plt.show()


# test d5 dataset
def test_d5():
    data_path = './data/Digit-Five'
    transform = transforms.Compose([
        # PIL Image: [H, W, C], range: [0, 255]
        transforms.Resize(32),
        # resized PIL Image, range: [0, 255]
        transforms.ToTensor(),
        # Tensor: [C, H, W], range: [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Normalized Tensor: [C, H, W]
    ])
    d5 = DigitFiveDataset(data_path, train=True, transform=transform)
    d5_loader = DataLoader(d5, batch_size=10, shuffle=True)

    # d5 loader (aligned dataset)
    for batch, data in enumerate(d5_loader):
        print('batch:', batch)
        # type(data): dict
        print(data['mnist']['image'].size())
        # data[DOMAIN]['image']: [N, C, H, W]
        print(data['mnist']['label'].size())
        # data[DOMAIN]['label']: [N,]


        # visualize image
        for img in data['mnist']['image']:
            # img: [C, H, W]
            visualize_img(img)

        break


# main
if __name__ == '__main__':
    # load_usps('./data/Digit-Five')
    test_d5()
