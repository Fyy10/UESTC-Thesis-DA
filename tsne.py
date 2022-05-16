from model import LeNet
from dataset import DigitFiveDataset

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_path = '/mnt/data/fyy'
checkpoint_path = './checkpoint/lenet_mnist_to_mnistm_batch128_adam_lr0.001_1.feat'
# num of samples to visualize for each domain
num_samples = 3000
source = 'mnist'
target = 'mnistm'

# d5 transform
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

d5_dataset = DigitFiveDataset(data_path=data_path, transform=transform)
d5_loader = DataLoader(d5_dataset, batch_size=num_samples, shuffle=True, num_workers=0)

data = next(iter(d5_loader))
# data: [num_samples, C, H, W]

data_in = torch.cat((data[source]['image'], data[target]['image']), dim=0)
# data_in: [2 * num_samples, C, H, W]

data_in = data_in.to(0)
feature_extractor = LeNet().to(0)
feature_extractor.load_state_dict(torch.load(checkpoint_path))

feat = feature_extractor(data_in)
# feat: [2 * num_samples, num_features]
# to numpy
feat = feat.cpu().detach().numpy()

# tsne feature embedding
tsne = TSNE(n_components=2)

# embed to 2 dimensions
feat_2 = tsne.fit_transform(feat)

plt.figure()
plt.scatter(feat_2[:num_samples, 0], feat_2[:num_samples, 1], c='b', s=5)
plt.scatter(feat_2[num_samples:, 0], feat_2[num_samples:, 1], c='r', s=5)
plt.axis('off')
# plt.xticks([])
# plt.yticks([])
plt.savefig('tsne.pdf', format='pdf')
