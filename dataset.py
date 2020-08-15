import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import torchvision.transforms.functional as ff
from augmentation import *
from torchvision import transforms
import torch

import xml.etree.ElementTree as ET

class LabelProcessor:

	def __init__(self):

		self.colormap = [(0, 0, 0),
		(255, 0, 0),
		(0, 255, 0),
		(255, 255, 0),
		(128, 128, 128),
		(0, 0, 255),
		(255, 0, 255),
		(0, 255, 255),
		(255, 255, 255)]

		self.color2label = self.encode_label_pix(self.colormap)

	@staticmethod
	def encode_label_pix(colormap):
		cm2lb = np.zeros(256**3)
		for i, cm in enumerate(colormap):
			cm2lb[(cm[0]*256 + cm[1]) * 256 + cm[2]] = i

		return cm2lb

	def encode_label_img(self, img):
		data = np.array(img, dtype='int32')
		idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
		label = np.array(self.color2label[idx], dtype='int64')

		return label


p = LabelProcessor()

class HeadSegData(data.Dataset):
	def __init__(self, datadir, trainxml, crop_size=(296, 280), train=True):
		self.datadir = datadir
		self.trainxml = trainxml
		self.crop_size = crop_size
		self.train = train

		self.src = []
		self.label = []

		#self.p = LabelProcessor()

		tree = ET.parse(self.trainxml)
		root = tree.getroot()

		if self.train:
			for i in range(0, len(root)):
				if root[i].tag == 'bboxes':
					continue

				if 'real' in root[i].attrib['name']:
					break

				if root[i].tag == 'srcimg':
					self.src.append(root[i].attrib['name'])

				if root[i].tag == 'labelimg':
					self.label.append(root[i].attrib['name'])


		else:
			for i in range(0, len(root)):
				if root[i].tag == 'bboxes':
					continue
				if 'real' in root[i].attrib['name']:

					if root[i].tag == 'srcimg':
						self.src.append(root[i].attrib['name'])

					if root[i].tag == 'labelimg':
						self.label.append(root[i].attrib['name'])

	def __getitem__(self, index):
		img_file = self.src[index]
		label_file = self.label[index]

		img_path = os.path.join(self.datadir, img_file).replace("\\","/")
		label_path = os.path.join(self.datadir, label_file).replace("\\","/")

		img = Image.open(img_path).convert("RGB")
		label = Image.open(label_path).convert("RGB")

		img, label = self.center_crop(img, label, self.crop_size)

		img, label, y_cls = self.img_transform(img, label, index)

		return img, label, y_cls
	
	def __len__(self):
		return len(self.src)


	def center_crop(self, img, label, crop_size):
		img = ff.center_crop(img, crop_size)
		label = ff.center_crop(label, crop_size)

		return img, label

	def img_transform(self, img, label, index):
		label = np.array(label)
		label = Image.fromarray(label.astype('uint8'))

		transform_label = transforms.Compose([
			transforms.ToTensor()]
			)

		transform_img = transforms.Compose(
			[
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
			)

		img = transform_img(img)

		label = p.encode_label_img(label)
        #Image.fromarray(label).save("./results_pic/"+str(index)+".png")
        #print(label.shape)
		y_cls, _ = np.histogram(label, bins=9, range=(-0.5, 9-0.5), )
		y_cls = np.asarray(np.asarray(y_cls, dtype=np.bool), dtype=np.uint8)

        #label = transform_label(label)
        #label = torch.squeeze(label)

		return img, torch.from_numpy(label), torch.from_numpy(y_cls)


