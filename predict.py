import os
import torch
import logging
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataset import HeadSegData
from PIL import Image
from pspnet import PSPNet

snapshot = "./models/9.pth"
data_path = './data/dataset'
xml_path = './data/dataset/training.xml'

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
    net = models[backend]().cuda(0)
    if snapshot is not None:
        epoch = os.path.basename(snapshot).split('.')[0]
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    return net.cuda(0), epoch

net, _ = build_network(snapshot, 'resnet34')

testdata = HeadSegData(data_path, xml_path, train=True)
test_loader = DataLoader(testdata, batch_size=1, shuffle=True, num_workers=1)
net.eval()

colormap = [[0, 0, 0],[255, 0, 0],[0, 255, 0],[255, 255, 0],[128, 128, 128],[0, 0, 255],[255, 0, 255],[0, 255, 255],[255, 255, 255]]

cm = np.array(colormap).astype('uint8')

dir = "./results_pic/"

with torch.no_grad():
	for i, (img, _, _) in tqdm(enumerate(test_loader)):
		img = img.cuda(0)
		out, _ = net(img)

		pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
		pre_label = np.asarray(pre_label, dtype=np.uint8)
		pre = cm[pre_label]
		pre1 = Image.fromarray(pre.astype("uint8"), mode='RGB')
		pre1.save(dir + str(i) + '.png')
		print("Done")
