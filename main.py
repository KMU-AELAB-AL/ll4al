import os
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import CIFAR100, CIFAR10

from tqdm import tqdm
from tensorboardX import SummaryWriter

from config import *
from models.resnet import ResNet18
from models.vq_vae import VQVAE
from data.transform import Cifar
from data.sampler import SubsetSequentialSampler


random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


transforms = Cifar()


if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)
else:
    raise FileExistsError


def train_backbone_epoch(models, criterions, optimizers, dataloaders):
    models['backbone'].train()
    models['uncertainty'].eval()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        loss = criterions['backbone'](scores, labels)

        loss.backward()
        optimizers['backbone'].step()


def train_uncertainty_epoch(models, criterions, optimizers, dataloaders, summary_writer, epoch):
    models['backbone'].eval()
    models['uncertainty'].train()

    cnt = 0
    _loss = 0.
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()

        optimizers['uncertainty'].zero_grad()

        vq_loss, recon, _, _ = models['backbone'](inputs)
        recon_loss = criterions['uncertainty'](recon, inputs)

        loss = vq_loss + (torch.sum(recon_loss) / recon_loss.size(0))

        loss.backward()
        optimizers['uncertainty'].step()

        _loss += loss

        summary_writer.add_image('image/origin', inputs[0], epoch)
        summary_writer.add_image('image/recon', recon[0], epoch)
        summary_writer.add_scalar('loss', _loss / cnt, epoch)

    return _loss / cnt


def test(models, dataloaders, mode='val'):
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(models, criterions, optimizers, schedulers, dataloaders, num_epochs):
    summary_writer = SummaryWriter(log_dir=os.path.join('./'), comment='VAE')

    print('>> Train a Model.')

    checkpoint_dir = os.path.join(f'./trained', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_backbone_epoch(models, criterions, optimizers, dataloaders)

    for epoch in range(num_epochs):
        loss = train_uncertainty_epoch(models, criterions, optimizers, dataloaders, summary_writer, epoch)
        schedulers['uncertainty'].step(loss)

    print('>> Finished.')


def get_uncertainty(models, criterions, unlabeled_loader):
    models['backbone'].eval()
    models['uncertainty'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            _, recon, _, _ = models['uncertainty'](inputs)
            loss = criterions['uncertainty'](inputs)

            uncertainty = torch.cat((uncertainty, loss), 0)
    
    return uncertainty.cpu()


if __name__ == '__main__':
    for trial in range(TRIALS):
        fp = open(f'record_{trial + 1}.txt', 'w')

        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:INIT_CNT]
        unlabeled_set = indices[INIT_CNT:]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        uncertainty_model = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, NUM_EMBEDDINGS,
                                  EMBEDDING_DIM, COMMITMENT_COST, DECAY).cuda()
        resnet18 = ResNet18(num_classes=CLS_CNT).cuda()
        models = {'backbone': resnet18, 'uncertainty': uncertainty}

        torch.backends.cudnn.benchmark = False

        for cycle in range(CYCLES):
            criterion_backbone = nn.CrossEntropyLoss().cuda()
            criterion_uncertainty = nn.MSELoss(reduction='none').cuda()
            criterions = {'backbone': criterion_backbone, 'uncertainty': criterion_uncertainty}

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optim_uncertainty = optim.Adam(models['uncertainty'].parameters())
            optimizers = {'backbone': optim_backbone, 'uncertainty': optim_uncertainty}

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_uncertainty = lr_scheduler.ReduceLROnPlateau(optim_uncertainty, mode='min', factor=0.8, cooldown=4)
            schedulers = {'backbone': sched_backbone, 'uncertainty': sched_uncertainty}

            train(models, criterions, optimizers, schedulers, dataloaders, EPOCH)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            uncertainty = get_uncertainty(models, criterions, unlabeled_loader)

            arg = np.argsort(uncertainty)

            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        fp.close()
