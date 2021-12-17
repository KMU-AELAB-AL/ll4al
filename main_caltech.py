import os
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from torchvision.datasets import Caltech101

from tqdm import tqdm

from config import *
from models.resnet import ResNet18
from models.lossnet import LossNet
from data.transform import Caltech
from data.sampler import SubsetSequentialSampler


random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


transforms = Caltech()
data_train = Caltech101('./data', download=True, transform=transforms.train_transform)
data_unlabeled = Caltech101('./data', download=True, transform=transforms.test_transform)
data_test = Caltech101('./data', download=True, transform=transforms.test_transform)


def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)
        loss = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()


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


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    checkpoint_dir = os.path.join(f'./trained', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


if __name__ == '__main__':
    for trial in range(TRIALS):
        fp = open(f'record_{trial + 1}.txt', 'w')

        indices, test_set = train_test_split(np.arange(len(data_train.y)), test_size=0.1,
                                             random_state=42, stratify=data_train.y)
        indices, test_set = list(indices), list(test_set)

        labeled_set = indices[:INIT_CNT]
        unlabeled_set = indices[INIT_CNT:]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH, sampler=SubsetRandomSampler(test_set))
        dataloaders = {'train': train_loader, 'test': test_loader}

        loss_module = LossNet().cuda()
        resnet18 = ResNet18(num_classes=CLS_CNT).cuda()
        models = {'backbone': resnet18, 'module': loss_module}

        torch.backends.cudnn.benchmark = False

        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone, 'module': optim_module}

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            uncertainty = get_uncertainty(models, unlabeled_loader)

            arg = np.argsort(uncertainty)

            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        fp.close()
