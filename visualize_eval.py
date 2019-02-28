import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import SequentialLoadingrate
import numpy as np
from matplotlib import pyplot as plt

def RMSE(x, y):
    return np.sqrt(np.sum(np.square(x - y))/len(x))

# training fc softmax classification
class fcNet(nn.Module):
    def __init__(self):
        super(fcNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(1000, 256),
                                nn.PReLU(),
                                nn.Linear(256, 101)
                                )
    def forward(self, x):
        output = x.view(x.size()[0], -1)
        output = self.fc(output)
        return output


# Hyper parameters
folder_str = 'reg4'
num_epochs = 50
batch_size = 1
learning_rate = 0.0001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose(transforms=[transforms.Resize(224, 224),
                                       transforms.ToTensor()])

test_dataset = SequentialLoadingrate('/media/zjs/A638551F3854F033/loadingrate/test/%s/loadingrate.txt' % folder_str,
                                   '/media/zjs/A638551F3854F033/loadingrate/test/%s' % folder_str,
                                   transform=trans)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# Test the model
basenet = models.resnet34(pretrained=False)
checkpoint = torch.load('./checkpoints/basenet_1_final.ckpt')
basenet.load_state_dict(checkpoint)
model = fcNet()
model.load_state_dict(torch.load('./checkpoints/model_1_final.ckpt'))
basenet.to(device)
model.to(device)
basenet.eval()
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    big_error = 0
    gt = []
    pred = []
    for (image, label) in test_loader:
        gt.append(int(label))
        image = image.to(device)
        label = label.to(device)
        output = model(basenet(image))
        _, predicted = torch.max(output.data, 1)
        if len(pred):
            predicted = (pred[-1] + predicted)/2
        pred.append(int(predicted))
        total += label.size(0)
        for i in range(len(label)):
            correct += int(abs(predicted[i] - label[i]) <= 10)
            big_error += int(abs(predicted[i] - label[i]) > 20)
        # correct += ((predicted - labels) <= 10).sum().item()

    rmse = RMSE(np.array(gt), np.array(pred))
    print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))
    print('Test error bigger than  20 on test images: {} %'.format(100 * big_error / total))
    print('Test RMSE     of the model on test images: {} '.format(rmse))

    l1, = plt.plot(range(len(test_loader)), gt, label='Truth')
    l2, = plt.plot(range(len(test_loader)), pred, 'r-', label='Predict')
    l3, = plt.plot(range(len(test_loader)), [abs(gt[i] - pred[i]) for i in range(len(gt))], label='diff', c='g', linestyle='--')
    l4, = plt.plot(range(len(test_loader)), [10.0] * len(gt), c='k', linestyle='-.', label='10')
    plt.legend(handles=[l1, l2, l3, l4], loc='lower left')
    plt.savefig('./images/test_%s.png' % folder_str, dpi=200)
    plt.show()
    # plt.close()