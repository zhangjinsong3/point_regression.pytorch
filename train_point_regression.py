import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 400
batch_size = 64
batch_size_test = 1
learning_rate = 0.0001

# Data loader
from datasets import BoxPoint, BoxPointFromSeven, BoxPointAsOne
trans = transforms.Compose(transforms=[transforms.Resize(96, 96),
                                       transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1),
                                       transforms.ToTensor()])
# train_dataset = BoxPoint('/media/zjs/A22A53E82A53B7CD/kuaice/data_mingjian/data/data_train.txt',
#                          '/media/zjs/A22A53E82A53B7CD/kuaice/data_mingjian/data/data_train',
#                          ignore=['0', '4', '5', '6'],
#                          transform=trans)
train_dataset = BoxPointFromSeven('../../data_mingjian/data/image_train/train.txt',
                                  '../../data_mingjian/data/image_train',
                                  transform=trans)
# train_dataset = BoxPointAsOne('../../data_mingjian/data/image_train/train_one.txt',
#                               '../../data_mingjian/data/image_train',
#                               transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

trans_test = transforms.Compose(transforms=[transforms.Resize(96, 96),
                                            transforms.ToTensor()])
# test_dataset = BoxPoint('/media/zjs/A22A53E82A53B7CD/kuaice/data_mingjian/data/data_test.txt',
#                         '/media/zjs/A22A53E82A53B7CD/kuaice/data_mingjian/data/data_test',
#                         ignore=['0', '4', '5', '6'],
#                         transform=trans_test)
test_dataset = BoxPointFromSeven('../../data_mingjian/data/image_test/test.txt',
                                 '../../data_mingjian/data/image_test',
                                 transform=trans_test)
# test_dataset = BoxPointAsOne('../../data_mingjian/data/image_test/test.txt',
#                              '../../data_mingjian/data/image_test',
#                              transform=trans_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=False)

basenet = models.resnet18(pretrained=False, num_classes=2)
# basenet.avgpool = nn.AdaptiveAvgPool2d(1)
basenet.avgpool = nn.AvgPool2d(7, stride=1)
# from networks import miniResNet
# basenet = miniResNet(models.resnet.BasicBlock, [2, 2, 2])
# from networks import PointNet
# basenet = PointNet()
from networks import MobileNet
basenet = MobileNet()

# resume training from checkpoint
# checkpoint = torch.load('./checkpoints/pointnet_all_0200.ckpt')
# basenet.load_state_dict(checkpoint)

# Loss and optimizer
criterion = nn.SmoothL1Loss()

# criterion = nn.MSELoss()
# from losses import TripletLoss
# criterion = TripletLoss(margin=1)
optimizer = torch.optim.Adam(basenet.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train the model
writer = SummaryWriter()
# show the net
# dummy_input = torch.rand(8, 3, 224, 224)
# with SummaryWriter(comment='resnet34') as w:
#     w.add_graph(basenet, dummy_input)
basenet = basenet.to(device)
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, point) in enumerate(train_loader):
        image = image.to(device)
        point = point.to(device)

        # Forward pass
        output1 = basenet(image)

        loss = criterion(output1, point)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show the loss curve
        writer.add_scalar('scalar/loss_0', loss.item(), epoch * total_step + i)
        # show the filter learned
        # for name, param in basenet.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch * total_step + i)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    if (epoch + 1) % 20 == 0:
        from eval_point_regression import eval_point_regression
        # Save the model checkpoint
        eval_point_regression(basenet, train_loader, batch_size=batch_size)
        eval_point_regression(basenet, test_loader, batch_size=batch_size_test)
        torch.save(basenet.state_dict(), 'checkpoints/mobilenet_0_%04d.pth' % (epoch+1))
        basenet.train()
print("box regression trainning finished! ")

writer.close()
