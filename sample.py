from model import LeNet, Classifier
from dataset import DigitFiveDataset

import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# reverse transform
def reverse_transform(img):
    """
    Args:
        img: image Tensor of size [C, H, W]

    Returns:
        img: PIL image
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

    return img


data_path = './data/Digit-Five'
checkpoint_path = './checkpoint/lenet_mnistm_to_mnist_batch128_adam_lr0.001_0'
feature_extractor_path = checkpoint_path + '.feat'
classifier_path = checkpoint_path + '.cls'

# num of samples to visualize for each domain
num_samples = 6
source = 'mnistm'
target = 'mnist'

# d5 transform
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

d5_dataset = DigitFiveDataset(data_path=data_path, transform=transform, train=False)
d5_loader = DataLoader(d5_dataset, batch_size=num_samples, shuffle=True, num_workers=0)

data = next(iter(d5_loader))
# data: [num_samples, C, H, W]

data_in = torch.cat((data[source]['image'], data[target]['image']), dim=0)
# data_in: [2 * num_samples, C, H, W]

data_in = data_in.to(0)
feature_extractor = LeNet().to(0)
feature_extractor.load_state_dict(torch.load(feature_extractor_path))
classifier = Classifier(in_dim=84, out_dim=10).to(0)
classifier.load_state_dict(torch.load(classifier_path))

feat = feature_extractor(data_in)
# feat: [2 * num_samples, num_features]
pred = classifier(feat)
pred = torch.sigmoid(pred)
# pred: [2 * num_samples, 10]
# to numpy
feat = feat.cpu().detach().numpy()
pred = pred.cpu().detach().numpy()

# source
fig, ax = plt.subplots(nrows=2, ncols=num_samples, num='Source_Pred')
fig.suptitle('Source', fontsize=20)
for idx, pair in enumerate(zip(data_in[:num_samples], pred[:num_samples])):
    img, p = pair
    # img: [C, H, W]
    # p: [10,]

    # plot image
    img = reverse_transform(img)
    ax[0, idx].imshow(img)
    ax[0, idx].axis('off')

    # plot prediction
    x = list(range(10))
    ax[1, idx].bar(x, p, color='c')
    ax[1, idx].set_ylim(bottom=0, top=1)
    ax[1, idx].set_xticks(x)
    if idx != 0:
        ax[1, idx].get_yaxis().set_visible(False)

fig.subplots_adjust(hspace=0.0, wspace=0.0)
plt.show()

# target
fig, ax = plt.subplots(nrows=2, ncols=num_samples, num='Target_Pred')
fig.suptitle('Target', fontsize=20)
for idx, pair in enumerate(zip(data_in[num_samples:], pred[num_samples:])):
    img, p = pair
    # img: [C, H, W]
    # p: [10,]

    # plot image
    img = reverse_transform(img)
    ax[0, idx].imshow(img)
    ax[0, idx].axis('off')

    # plot prediction
    x = list(range(10))
    ax[1, idx].bar(x, p, color='tab:brown')
    ax[1, idx].set_ylim(bottom=0, top=1)
    ax[1, idx].set_xticks(x)
    if idx != 0:
        ax[1, idx].get_yaxis().set_visible(False)

fig.subplots_adjust(hspace=0.0, wspace=0.0)
plt.show()
