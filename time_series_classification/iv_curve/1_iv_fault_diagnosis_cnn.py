# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/1 19:16
# @Author : liumin
# @File : 1_iv_fault_diagnosis_cnn.py
import copy
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

import matplotlib
import cv2
import glob


def channel_shuffle1d(x, groups):
    batchsize, num_channels, length = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, length)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, length)

    return x

class InvertedResidual1D(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv1d(inp, inp, 3, self.stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.Conv1d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv1d(inp if (self.stride > 1) else branch_features,
                branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_features, branch_features, 3, self.stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.Conv1d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle1d(out, 2)
        return out


class IVModel(nn.Module):
    def __init__(self, inp=2, oup=2):
        super(IVModel,self).__init__()
        blocks = [2, 4, 2]
        channels =[12, 24, 48, 96, 196]
        self.stem = nn.Sequential(
            nn.Conv1d(inp, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for i, (name, block) in enumerate(zip(stage_names, blocks)):
            seq = [InvertedResidual1D(channels[i], channels[i+1], 2)]
            for _ in range(block - 1):
                seq.append(InvertedResidual1D(channels[i+1], channels[i+1], 1))
            setattr(self, name, nn.Sequential(*seq))

        self.lastconv = nn.Sequential(
            nn.Conv1d(channels[3], channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm1d(channels[4]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(channels[4], oup)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.lastconv(x)
        x = x.mean([2])  # globalpool
        x = self.fc(x)
        return x

'''
model = IVModel()
print(model)
input = torch.randn(1, 2, 100)
out = model(input)
'''

class IVDataset(Dataset):
    def __init__(self, root_path='', stage='train', input_size=100, p=0.8):
        super(IVDataset, self).__init__()
        self.root_path = root_path
        self.stage = stage
        self.input_size = input_size
        self.items = []
        self.labels = []
        self.names = []

        with open(self.root_path, 'rb') as f:
            iv_data = pickle.load(f)

        if self.stage == 'test':
            for k, v in iv_data.items():
                self.items.append(np.array([v['pvv'], v['pvi']]))  # 2*100
        else:
            for k, v in iv_data.items():
                self.items.append(np.array([v['pvv'], v['pvi']])) # 2*100
                self.labels.append(v['lbl'])
        print('{} :datasets: {}'.format(self.stage, len(self.items)))


    def __getitem__(self, idx):
        item = self.items[idx]

        if self.stage == "train":
            item = torch.from_numpy(item)
            return item, self.labels[idx]
        else:
            item = torch.from_numpy(item)
            if self.stage == "test":
                return item
            else:
                return item, self.labels[idx]

    def __len__(self):
        return self.items.__len__()


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, device):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class TrainIVcurve():
    def __init__(self):
        super(TrainIVcurve, self).__init__()
        self.data_dir = {'train':'iv_data_train.txt', 'val':'iv_data_val.txt'}
        # self.data_dir = {'train':'iv_data.txt', 'val':'iv_data.txt'}
        self.pretrain_model_path = f'weights/iv_data_weight.pth'
        self.save_model_path = f'weights/iv_data_weight.pth'
        self.input_size = 100 # length
        self.output_num = 2

        self.batch_size = 64
        self.num_epochs = 500  #
        self.init_lr = 0.01  # [0.1 0.01 0.001 0.0001 0.00001 0.000001]0.01 0.001 0.0001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def run(self):
        image_datasets = {x: IVDataset(root_path=self.data_dir[x], stage=x, input_size=self.input_size) for x in ['train', 'val']}  # , x
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True if x == 'train' else False,
                          num_workers=8) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        model_ft = IVModel()
        if self.pretrain_model_path is not None and os.path.exists(self.pretrain_model_path):
            pass
            # print('load pretrained model: ', self.pretrain_model_path)
            # model_ft.load_state_dict(torch.load(self.pretrain_model_path), strict=False)
        model_ft = model_ft.to(self.device)

        criterion = FocalLoss(class_num=self.output_num)
        criterion.cuda(self.device)

        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) # , weight_decay=0.0005
        # lr_scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, self.num_epochs, eta_min=1e-7, last_epoch=-1)

        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.995, 0.999)) # weight_decay=0.0005
        lr_scheduler_ft = None

        since = time.time()
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0
        preds_list = []
        gt_labels_list = []
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 50)
            preds_list.clear()
            gt_labels_list.clear()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_ft.train()  # Set model to training mode
                else:
                    model_ft.eval()  # Set model to evaluate mode
                    preds_list = []
                    gt_labels_list = []

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)

                        # loss = criterion(outputs, labels)
                        loss = criterion(outputs, labels, self.device)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()
                            if lr_scheduler_ft is not None:
                                lr_scheduler_ft.step()
                    preds_list.extend(preds.cpu().detach().numpy().tolist())
                    gt_labels_list.extend(labels.data.cpu().detach().numpy().tolist())
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
            print('')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model_ft.load_state_dict(best_model_wts)
        torch.save(model_ft.state_dict(), self.save_model_path)

if __name__ == '__main__':
    t = TrainIVcurve()
    t.run()
    print('finished!')

