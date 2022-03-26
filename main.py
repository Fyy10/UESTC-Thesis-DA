import model.CNN as CNN
import model.ViT as ViT
# NOTE: do not use aligned d5 dataset
from dataset.DigitFive import DigitFiveDataset
from metric.k_moment import KMomentLoss
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
from torchvision import transforms


# training settings
# basic description
parser = argparse.ArgumentParser(description='A model for Domain Adaptation')
# select model
parser.add_argument(
    '--model',
    type=str, metavar='str',
    default='lenet',
    help='select model (lenet, alexnet, alexnet_org, resnet50, vit_b16, default: lenet)'
)
# record folder
parser.add_argument(
    '--record_folder',
    type=str, metavar='str',
    default='record',
    help='record folder (default: record)'
)
# batch size
parser.add_argument(
    '--batch_size',
    type=int, metavar='int',
    default=128,
    help='batch size for training (default: 128)'
)
# number of epochs
parser.add_argument(
    '--num_epoch',
    type=int, metavar='int',
    default=10,
    help='number of epochs for training (default: 10)'
)
# checkpoint folder
parser.add_argument(
    '--checkpoint',
    type=str, metavar='str',
    default='checkpoint',
    help='checkpoint folder (default: checkpoint)'
)
# load a model (provede the filename)
parser.add_argument(
    '--load',
    type=str, metavar='str',
    default=None,
    help='load pretrained model, provide the filename (default: None)'
)
# random seed
parser.add_argument('--seed', type=int, metavar='int', default=1, help='random seed (default: 1)')
# dataset
parser.add_argument(
    '--dataset',
    type=str, metavar='str',
    default='d5',
    help='select dataset (d5, o31, dn, default: d5)'
)
# source domain
parser.add_argument('--source', type=str, metavar='str', default='svhn', help='select source domain (default: svhn)')
# target domain
parser.add_argument('--target', type=str, metavar='str', default='mnistm', help='select target domain (default: mnistm)')
# learning rate
parser.add_argument('--lr', type=float, metavar='float', default=1e-3, help='learning rate (default: 1e-3)')
# optimizer
parser.add_argument('--optim', type=str, metavar='str', default='adam', help='optimizer (default: adam)')
# gpu option
parser.add_argument('--use_cuda', action='store_true', default=False, help='use gpu (default: False)')
# select gpu
parser.add_argument('--gpu_num', type=int, metavar='int', default=0, help='select a gpu (default: 0)')
# save model
parser.add_argument('--save', action='store_true', default=False, help='save model or not (default: False)')
# save features for visualize
parser.add_argument('--vis', action='store_true', default=False, help='save features for visualize (default: False)')
# evaluate without training
parser.add_argument('--eval', action='store_true', default=False, help='evaluate without training (default: False)')
# gradient accumulation steps
parser.add_argument('--grad_step', type=int, metavar='int', default=1, help='number of gradient accumulation steps (default: 1)')
# distributed training and evaluating
parser.add_argument('--multi_card', action='store_true', default=False, help='distributed training and evaluating (default: False)')


# main
def main():
    # parse args
    args = parser.parse_args()

    # use cuda or not
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.cuda:
        args.device = torch.device(args.gpu_num)
    else:
        args.device = torch.device('cpu')

    # checkpoint folder
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    # record folder
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)

    print(args)

    # check valid model
    if args.model not in ['lenet', 'alexnet', 'alexnet_org', 'resnet50', 'vit']:
        raise ValueError(args.model + ' is not a valid model')
    # check valid dataset
    if args.dataset not in ['d5', 'o31', 'dn']:
        raise ValueError(args.dataset + ' is not a valid dataset')

    # check valid domain
    # d5
    if args.dataset == 'd5':
        if args.source not in ['mnist', 'mnistm', 'svhn', 'syn', 'usps']:
            raise ValueError(args.source + ' is not a valid domain for dataset ' + args.dataset)
        if args.target not in ['mnist', 'mnistm', 'svhn', 'syn', 'usps']:
            raise ValueError(args.target + ' is not a valid domain for dataset ' + args.dataset)
    # o31
    if args.dataset == 'o31':
        if args.source not in ['A', 'D', 'W']:
            raise ValueError(args.source + ' is not a valid domain for dataset ' + args.dataset)
        if args.target not in ['A', 'D', 'W']:
            raise ValueError(args.target + ' is not a valid domain for dataset ' + args.dataset)
    # dn
    if args.dataset == 'dn':
        if args.source not in ['clp', 'info', 'pnt', 'qdr', 'real', 'skt']:
            raise ValueError(args.source + ' is not a valid domain for dataset ' + args.dataset)
        if args.target not in ['clp', 'info', 'pnt', 'qdr', 'real', 'skt']:
            raise ValueError(args.target + ' is not a valid domain for dataset ' + args.dataset)

    # dataset-specific transform
    transform = None
    if args.dataset == 'd5':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.dataset == 'o31':
        num_classes = 31
        transform = transforms.Compose([
            # TODO: transform for o31
        ])
    if args.dataset == 'dn':
        num_classes = 345
        transform = transforms.Compose([
            # TODO: transform for dn
        ])

    # load dataset
    train_dataset = None
    test_dataset = None
    if args.dataset == 'd5':
        train_dataset = DigitFiveDataset('./data/Digit-Five', train=True, transform=transform)
        test_dataset = DigitFiveDataset('./data/Digit-Five', train=False, transform=transform)
    # TODO: load other datasets

    # dataloader
    num_workers = 0 if sys.platform.startswith('win') else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    # model
    feature_extractor = None
    classifier = None
    if args.model == 'lenet':
        feature_extractor = CNN.LeNet().to(args.device)
        classifier = CNN.Classifier(in_dim=84, out_dim=num_classes).to(args.device)
    # TODO: complete other models

    # load model if specified
    if args.load is not None:
        feature_extractor.load_state_dict(torch.load(args.load + '.feat'))
        classifier.load_state_dict(torch.load(args.load + '.cls'))

    if not args.eval:
        train(args, feature_extractor, classifier, train_loader)
    else:
        evaluate(args, feature_extractor, classifier, test_loader, args.target)


# train
def train(args, feature_extractor, classifier, dataloader):
    """
    Train the model.

    Args:
        args: argument parse
        feature_extractor: feature extraction network
        classifier: classification network
        dataloader: DataLoader
    """
    # experiment name (model, source, target, batch, optim, lr)
    exp_name = args.model + '_' + args.source + '_to_' + args.target + '_batch' + str(args.batch_size) + '_' + args.optim + '_lr' + str(args.lr)
    # experiment id (num of repeated experiments)
    exp_id = 0

    # find previous records
    while os.path.exists(os.path.join(args.record_folder, exp_name + '_' + str(exp_id))):
        exp_id += 1

    # make new records
    # experiment name with id
    exp_name_id = exp_name + '_' + str(exp_id)

    # record file
    # record_file = open(os.path.join(args.record_folder, exp_name_id + '.log'), 'w')

    # write args to record
    # print(args, file=record_file)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(args.record_folder, exp_name_id))
    sample_data = next(iter(dataloader))
    writer.add_graph(feature_extractor, input_to_model=sample_data[args.source]['image'].to(args.device))
    writer.add_text(exp_name_id, str(args.__dict__))

    # loss function
    cross_entropy = nn.CrossEntropyLoss()
    k_moment = KMomentLoss(k=4)

    # optimizer
    optim_f = None
    optim_c = None
    if args.optim == 'adam':
        optim_f = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr)
        optim_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    if args.optim == 'adamw':
        optim_f = torch.optim.AdamW(feature_extractor.parameters(), lr=args.lr)
        optim_c = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    if args.optim == 'sgd':
        optim_f = torch.optim.SGD(feature_extractor.parameters(), lr=args.lr)
        optim_c = torch.optim.SGD(classifier.parameters(), lr=args.lr)
    # TODO: complete other optimizers

    n_iter = 0
    best_acc = 0
    for epoch in range(args.num_epoch):
        print('Epoch:', epoch)
        for batch, data in enumerate(tqdm(dataloader)):
            # data[DOMAIN]['image']: [N, C, H, W]
            # data[DOMAIN]['label']: [N,]
            image_source = data[args.source]['image'].to(args.device)
            label_source = data[args.source]['label'].type(torch.LongTensor).to(args.device)
            image_target = data[args.target]['image'].to(args.device)
            label_target = data[args.target]['label'].type(torch.LongTensor).to(args.device)

            image = torch.cat((image_source, image_target), dim=0)
            label = torch.cat((label_source, label_target), dim=0)
            # image: [2 * N, C, H, W]
            # label: [2 * N,]

            # zero grad
            optim_f.zero_grad()
            optim_c.zero_grad()

            # forward
            feature = feature_extractor(image)
            # prediction
            pred = classifier(feature)

            feature_source, feature_target = torch.chunk(feature, 2, dim=0)
            pred_source, pred_target = torch.chunk(pred, 2, dim=0)
            # feature_source, feature_target: [N, num_features]
            # pred_source, pred_target: [N, num_classes]

            # predicted source labels
            _, pred_source_labels = pred_source.detach().topk(1, dim=1)
            pred_source_labels = pred_source_labels.squeeze(dim=1)
            # source accuracy
            batch_acc = (pred_source_labels == label_source).sum().item() / len(label_source)

            loss_ce = cross_entropy(pred_source, label_source)
            loss_km = k_moment(feature_source, feature_target)
            loss = loss_ce + 1e-2 * loss_km

            # write loss and accuracy
            writer.add_scalar('train/' + args.source + '_to_' + args.target + '_loss', loss.item(), n_iter)
            writer.add_scalar('train/' + args.source + '_to_' + args.target + '_acc', batch_acc, n_iter)

            # backward
            loss.backward()

            # optimizer step
            optim_f.step()
            optim_c.step()

            n_iter += 1

        acc_source = evaluate(args, feature_extractor, classifier, dataloader, args.source)
        writer.add_scalar('eval/' + args.source + '_to_' + args.target + '_acc_source', acc_source, epoch)
        acc_target = evaluate(args, feature_extractor, classifier, dataloader, args.target)
        writer.add_scalar('eval/' + args.source + '_to_' + args.target + '_acc_target', acc_target, epoch)

        # update best_acc (should only use source acc)
        if acc_source > best_acc:
            best_acc = acc_source
            # save model if specified
            if args.save:
                torch.save(feature_extractor.state_dict(), os.path.join(args.checkpoint, exp_name_id + '.feat'))
                torch.save(classifier.state_dict(), os.path.join(args.checkpoint, exp_name_id + '.cls'))

    # close record file
    # record_file.close()
    # close writer
    writer.close()


# evaluate
def evaluate(args, feature_extractor, classifier, dataloader, domain=None):
    """
    Evaluate the model.

    Args:
        args: argument parse
        feature_extractor: feature extraction network
        classifier: classification network
        dataloader: DataLoader
        domain: domain to be evaluated (str)
    """
    print('Evaluating...')

    if domain is None:
        domain = args.source

    print('Domain:', domain)
    acc = 0
    cnt = 0
    for batch, data in enumerate(tqdm(dataloader)):
        # data[DOMAIN]['image']: [N, C, H, W]
        # data[DOMAIN]['label']: [N,]
        image = data[domain]['image'].to(args.device)
        label = data[domain]['label'].to(args.device)

        cnt += image.size(0)

        feature = feature_extractor(image)
        # feature: [N, num_features]
        pred = classifier(feature)
        # pred: [N, num_classes]

        _, pred_labels = pred.topk(1, dim=1)
        pred_labels = pred_labels.squeeze(dim=1)
        # pred_labels: [N,]

        acc += (pred_labels == label).sum().item()

    acc /= cnt
    print('Accuracy on {}: {}'.format(domain, acc))

    return acc

# main
if __name__ == '__main__':
    main()
