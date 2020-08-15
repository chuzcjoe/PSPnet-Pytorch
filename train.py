import os
import logging
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from pspnet import PSPNet


models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    #net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda(0)
    return net, epoch


@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder', default='./data/dataset/')
@click.option('--trainxml', type=str, help='Path to xml file', default='./data/dataset/training.xml')
@click.option('--models-path', type=str, help='Path for storing model snapshots', default='./models')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=8)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

def train(data_path, trainxml, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    #net, starting_epoch = build_network(snapshot, backend)
    #data_path = os.path.abspath(os.path.expanduser(data_path))
    #models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    traindata = HeadSegData(data_path, trainxml, train=True)
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=1)

    net,_ = build_network(None, backend)
    seg_criterion = nn.NLLLoss().cuda(0)
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    #scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    print("start training...")
    net.train()
    for epoch in range(epochs):
    	if epoch % 6 == 0 and epoch != 0:
    		for group in optimizer.param_groups:
    			group['lr'] *= 0.5

    	for i, (x, y, y_cls) in enumerate(train_loader):
            x, y, y_cls = x.cuda(0), y.cuda(0).long(), y_cls.cuda(0).float()

            out, out_cls = net(x)
            seg_loss = seg_criterion(out, y)
            cls_loss = cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha*cls_loss

            if i % 50 == 0:
                status = '[batch:{0}/{1} epoch:{2}] loss = {3:0.5f}'.format(i, len(traindata)//batch_size, epoch + 1, loss.item())
                print(status)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    	torch.save(net.state_dict(), os.path.join(models_path, str(epoch)+".pth"))

    			  
if __name__ == '__main__':
    train()
