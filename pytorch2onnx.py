"""
convert pytorch 0.4.1 to onnx for ncnn use
created by zjs 2019.01.03
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import SequentialLoadingrate
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch.onnx

from torch.autograd import Variable

dummy_input = Variable(torch.randn(1, 3, 96, 96))

# model = models.resnet18(pretrained=False, num_classes=2)
# model.avgpool = nn.AvgPool2d(3, stride=1)
from networks import MobileNet
model = MobileNet()
model.load_state_dict(torch.load('./checkpoints/mobilenet_all_0050.pth'))

torch.onnx.export(model, dummy_input, "onnx/mobilenet_all.onnx")