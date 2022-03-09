from torch.utils.data import Dataset
import os
import numpy as np
from scipy.io import loadmat


# load mnist data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_mnist(data_dir, size=28):
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
    train_data = np.reshape(data['train_' + str(size)], (55000, size, size, 1)).transpose((0, 3, 1, 2))
    # train_data: [55000, 1, size, size]
    test_data = np.reshape(data['test_' + str(size)], (10000, size, size, 1)).transpose((0, 3, 1, 2))
    # test_data: [10000, 1, size, size]

    # labels are 0, 1, 2, ..., 9
    train_label = np.nonzero(data['label_train'])[1]
    # train_label: [55000,]
    test_label = np.nonzero(data['label_test'])[1]
    # test_label: [10000,]

    return train_data, test_data, train_label, test_label


# load mnistm data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_mnistm(data_dir):
    # mnistm data mat path
    file_path = os.path.join(data_dir, 'mnistm_with_label.mat')
    data = loadmat(file_path)
    # type(data): dict
    # keys: 'label_test': [10000, 10], 'label_train': [550000, 10], 'test': [10000, 28, 28, 3], 'train': [55000, 28, 28, 3]

    # value range: [0, 255]
    train_data = np.array(data['train']).transpose((0, 3, 1, 2))
    # train_data: [55000, 3, 28, 28]
    test_data = np.array(data['test']).transpose((0, 3, 1, 2))
    # test_data: [10000, 3, 28, 28]

    # labels are 0, 1, 2, ..., 9
    train_label = np.nonzero(data['label_train'])[1]
    # train_label: [55000,]
    test_label = np.nonzero(data['label_test'])[1]
    # test_label: [10000,]

    return train_data, test_data, train_label, test_label


# load svhn data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_svhn(data_dir):
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
    train_data_arr = np.array(train_data['X']).transpose((3, 2, 0, 1))
    # train_data_arr: [73257, 3, 32, 32]
    test_data_arr = np.array(test_data['X']).transpose((3, 2, 0, 1))
    # test_data: [26032, 3, 32, 32]

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
    train_data = np.array(data['train_data']).transpose((0, 3, 1, 2))
    # train_data: [25000, 3, 32, 32]
    test_data = np.array(data['test_data']).transpose((0, 3, 1, 2))
    # test_data: [9000, 3, 32, 32]

    # labels are 0, 1, 2, ..., 9
    train_label = np.array(data['train_label'])[:, 0]
    # train_label: [25000,]
    test_label = np.array(data['test_label'])[:, 0]
    # test_label: [9000,]

    return train_data, test_data, train_label, test_label


# load usps data
# refer to https://github.com/KaiyangZhou/Dassl.pytorch
def load_usps(data_dir):
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
    train_data = np.array(data['dataset'][0][0] * 255).astype(np.uint8)
    # train_data: [7438, 1, 28, 28]
    test_data = np.array(data['dataset'][1][0] * 255).astype(np.uint8)
    # test_data: [1860, 1, 28, 28]

    # labels are 0, 1, 2, ..., 9
    train_label = np.array(data['dataset'][0][1])[:, 0]
    # train_label: [7438,]
    test_label = np.array(data['dataset'][1][1])[:, 0]
    # test_label: [1860,]

    return train_data, test_data, train_label, test_label


# implement __init__(), __len__(), and __getitem__() for customized datasets
# TODO: DigitFiveDataset
class DigitFiveDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None) -> None:
        self.data = 0


# test d5 dataset
def test_d5():
    data_path = './data/Digit-Five'
    d5 = DigitFiveDataset(data_path)


# main
if __name__ == '__main__':
    # test_d5()
    load_usps('./data/Digit-Five')
