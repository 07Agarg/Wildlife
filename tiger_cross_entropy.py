from __future__ import print_function
import os
# import nni
import time
import pdb
import torch
import logging
import argparse
import torchvision
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import wandb
import shutil
#from birds_get_tree_target_2 import *
# from inat_get_target_tree import */
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger('tigers')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed) 

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BATCH_SIZE = 16
Hiden_Number = 512
lr = 0.1
nb_epoch = 100
criterion = nn.CrossEntropyLoss()
num_classes = [2]        # 3-level setup

config = {
    'lr': lr,
    'RANDOM_STATE' : 0,
    'BATCH_SIZE' : BATCH_SIZE,
    'Technique' : 'cross-entropy',
    'Model' : 'resnet18',
    'train transforms' : 'RandomHorizontalFlip',
    'val transforms' : '-',
    'optimizer' : 'SGD',
    'Dataset' : 'tigers',
    'epochs' : nb_epoch,
    'feature_size' : Hiden_Number,
}

wandb.init(project='tigers', entity='wildlife', config=config)
path_name = wandb.run.name
datasets_path = "/home/ashimag/Datasets/wildlife_datasets/tigers/"
print(path_name)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if not os.path.isdir(path_name):
    mkdir_p(path_name)

img_resolution = 448
mean_inat19 = [0.454, 0.474, 0.367]
std_inat19 = [0.237, 0.230, 0.249]
#Data
print('==> Preparing data..')

## Transforms used in making-better-mistakes.
transform_train = transforms.Compose([
                # transforms.Resize((img_resolution, img_resolution)),
                transforms.RandomResizedCrop(img_resolution),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_inat19, std_inat19)])
transform_test = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean_inat19, std_inat19)
])

trainset    = torchvision.datasets.ImageFolder(root='/home/ashimag/Datasets/wildlife_dataset/tigers/splits/train_images', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True)      # pin_memory variable =True, workers=2 in making-better-mistakes

valset = torchvision.datasets.ImageFolder(root='/home/ashimag/Datasets/wildlife_dataset/tigers/splits/val_images', transform=transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True)

testset = torchvision.datasets.ImageFolder(root='/home/ashimag/Datasets/wildlife_dataset/tigers/splits/test_images', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True)

print('==> Building model..')
# net = models.resnet50(pretrained=True)                  # resnet18 in mbm (making better mistakes)
net = models.resnet18(pretrained=True)

class model_bn(nn.Module):
    def __init__(self, model, feature_size=512, classes_num=num_classes[0]):

        super(model_bn, self).__init__() 

        self.features_2 =  nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 2048 * 1 * 1                  # Used for resnet18       448*448 image size.
        # self.num_ftrs = 2048 * 1 * 1               # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)
 
    def forward(self, x, targets):
        # pdb.set_trace()
        x = self.features_2(x)   
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512
        species_input = x
        species_out = self.classifier_3(species_input)
        ce_loss_species = criterion(species_out, targets)
        ce_loss =  ce_loss_species 

        return ce_loss, [species_out, targets]

use_cuda = torch.cuda.is_available()
net = model_bn(net, Hiden_Number, num_classes[0])

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

def train(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    species_correct = 0
    species_total= 0

    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        ce_loss,\
        [species_out, species_targets]= net(inputs, targets)

        loss = ce_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, species_predicted = torch.max(species_out.data, 1)
        species_total += species_targets.size(0)
        species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()

        train_species_acc_iter = 100.*(species_correct/species_total)
        if batch_idx % 50 == 0:
            print('Iteration %d, train_species_acc = %.5f, train_loss = %.6f' % \
                      (idx, train_species_acc_iter, loss.item()))

    train_species_acc = 100.*species_correct/species_total

    train_loss = train_loss/(idx+1) 
    wandb.log({
        'train_epoch_acc/species' : train_species_acc}, step=epoch)
    wandb.log({
        'train_epoch_ce_loss/species' : train_loss}, step=epoch)
    print('Epoch %d, train_species_acc = %.5f, train_loss = %.6f' % \
                          (epoch, train_species_acc, train_loss))
    return train_species_acc, train_loss

def test(epoch, net, testloader, optimizer, test_or_val):

    net.eval()
    test_loss = 0
    species_correct = 0
    species_total= 0

    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            ce_loss,\
            [species_out, species_targets] = net(inputs, targets)

            test_loss += ce_loss.item()

            _, species_predicted = torch.max(species_out.data, 1)
            species_total += species_targets.size(0)
            species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()

    test_species_acc = 100.*species_correct/species_total

    test_loss = test_loss/(idx+1)
    if test_or_val == "test":
        print('Iteration %d, test_species_acc = %.5f, test_loss = %.6f' % \
                              (epoch, test_species_acc, test_loss))
    elif test_or_val == "val":
        print('Iteration %d, val_species_acc = %.5f, val_loss = %.6f' % \
                              (epoch, test_species_acc, test_loss))
    return test_species_acc, test_loss

def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)

def save_checkpoint(state, filename='model_best.pth.tar'):
    checkpoint=path_name + "/"
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

optimizer = optim.SGD([
                        {'params': net.classifier_3.parameters(), 'lr': 0.1},
                        {'params': net.features_1.parameters(),   'lr': 0.1},
                         {'params': net.features_2.parameters(),   'lr': 0.01},
                     ], 
                      momentum=0.9, weight_decay=5e-4)

# optimizer = optim.Adam([
#                         {'params': net.classifier_3.parameters(), 'lr': 0.1},
#                         {'params': net.features_1.parameters(),   'lr': 0.1},
#                         {'params': net.features_2.parameters(),   'lr': 0.01},
#                      ], weight_decay=5e-4)


if __name__ == '__main__':
    try:
        max_val_acc = 0
        test_acc_to_max_val = 0
        best_epoch = 0
        for epoch in range(nb_epoch):

            optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch) / 10
            optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch) / 10
            optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch) / 100

            train_species_acc, train_loss = train(epoch, net, trainloader, optimizer)
            val_species_acc, val_loss = test(epoch, net, valloader, optimizer, "val")
            test_species_acc, test_loss = test(epoch, net, testloader, optimizer, "test")
            print("Val loss: %.3f, Test loss: %.3f" % (val_loss, test_loss))
            if val_species_acc > max_val_acc:
                max_val_acc = val_species_acc
                best_epoch = epoch
                test_acc_to_max_val = test_species_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': max_val_acc,
                'val_acc': val_species_acc,
                'test_acc': test_acc_to_max_val,
                'optimizer' : optimizer.state_dict(),
            }, filename='checkpoint.pth.tar')
            wandb.log({'val_ce_loss' : val_loss}, step=epoch)
            wandb.log({
                'total train ce loss' : train_loss,
                'train acc' : train_species_acc,
                'val acc' : val_species_acc,
            }, step=epoch)
            
    except Exception as exception:
        logger.exception(exception)
        raise
