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
		y_cls = set(list(np.reshape(label, (-1))))

		return label, y_cls

class HeadSegData(data.Dataset):
	def __init__(self, datadir, trainxml, crop_size=(296, 280), train=True):
		self.datadir = datadir
		self.trainxml = trainxml
		self.resize = resize

		self.src = []
		self.label = []
		self.y_cls = np.zeros(9, dtype='int64')

		self.p = LabelProcessor()

		tree = ET.parse('sampleset.xml')
		root = tree.getroot()

		if train:
			for i in range(0, len(root)):
				if 'real' in root[i].attrib['name']:
					break

				if root[i].tag == 'srcimg':
					self.src.append(root[i].attrib['name'])

				if root[i].tag == 'labelimg':
					self.label.append(root[i].attrib['name'])


		else:
			for i in range(0, len(root)):
				if 'real' in root[i].attrib['name']:

					if root[i].tag == 'srcimg':
						self.src.append(root[i].attrib['name'])

					if root[i].tag == 'labelimg':
						self.label.append(root[i].attrib['name'])

	def __getitem__(self, index):
		img_file = self.src[index]
		label_file = self.label[index]

		img = Image.open(img_file)
		label = Image.open(label_file)

		img, label = center_crop(img, label, self.crop_size)

		img, label, y_cls = self.img_transform(img, label)

		for cls in y_cls:
			self.y_cls[cls] = 1

		return img, label, torch.from_numpy(self.y_cls)


	def center_crop(self, img, label, crop_size):
		img = ff.center_crop(img, crop_size)
		label = ff.center_crop(img, crop_size)

		return img, label

	def img_transform(self, img, label):
		label = np.array(label)
		label = Image.fromarray(label.astype('uint8'))

		transform_all = transforms.Compose(
			[
			transforms.RandomAffine(degrees=0, scale=(0.5,2)),
			RandomHorizontalFlip(),
			RandomRotation()
			]
			)

		transform_img = transforms.Compose(
			[
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
			)

		img, label = transform_all(img, label)

		img = transform_img(img)
		label, y_cls = self.p.encode_label_img(label)
		label = torch.from_numpy(label)

		return img, label, y_cls


