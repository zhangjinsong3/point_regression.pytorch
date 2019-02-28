import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import SequentialLoadingrate
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def euclidean_distance(p1, p2):
    sq = np.square(p1-p2)
    sum = sq.mean(axis=1) * 2
    dis = np.sqrt(sum)
    return dis


def euclidean_distance_7(p1, p2):
    sq = np.square(p1-p2)
    sq = sq.reshape(-1, 2)
    sum = sq.mean(axis=1) * 2
    dis = np.sqrt(sum)
    return dis


def eval_point_regression(model, test_loader, batch_size=1, visualize=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    basenet = model
    basenet.to(device)
    basenet.eval()
    bad_num = 0
    with torch.no_grad():
        dis_all = []
        for i, (image, point) in enumerate(test_loader):
            if point[0][0] < 0 or point[0][1] < 0:
                if not (i+1) % 7:
                    print('-1 -1', end='\n')
                else:
                    print('-1 -1', end=' ')
                continue
            image = image.to(device)
            point = point.to(device)

            # Forward pass
            output = basenet(image)
            dis = euclidean_distance(output.cpu().numpy(), point.cpu().numpy())
            if len(dis) == batch_size:
                dis_all.append(dis)

            # if not (i + 1) % 7:
            #     print(output[0][0].cpu().numpy(), output[0][1].cpu().numpy(), end='\n')
            # else:
            #     print(output[0][0].cpu().numpy(), output[0][1].cpu().numpy(), end=' ')
            # print(np.array(dis), end=' ')
            # visualize one point
            if visualize:
                for k in range(dis.shape[0]):
                    if dis[k] <= 0:
                        continue
                    np_img = image.cpu().numpy()[k] * 255
                    np_img = np_img.astype('uint8').transpose((1, 2, 0)).copy()
                    cv2.circle(np_img, tuple((point[k].cpu().numpy()).astype('uint8')), 3, (0, 0, 255), -1)
                    cv2.circle(np_img, tuple((output[k].cpu().numpy()).astype('uint8')), 3, (0, 255, 0), -1)
                    cv2.namedWindow('seesee')
                    cv2.imshow('seesee', np_img)
                    cv2.waitKey()

    print('total mean distance :', np.mean(np.array(dis_all)).mean().mean())
    # print('each point distance error: ', np.array(dis_all).mean(axis=0))


if __name__ == '__main__':
    # Hyper parameters
    batch_size = 1

    trans = transforms.Compose(transforms=[transforms.Resize(96, 96),
                                           transforms.ToTensor()])
    # from datasets import BoxPoint
    # test_dataset = BoxPoint('../../data_mingjian/data/data_test.txt',
    #                         '../../data_mingjian/data/data_test',
    #                         # ignore=['4', '5', '6'],
    #                         transform=trans)

    # from datasets import BoxPointFromSeven
    # test_dataset = BoxPointFromSeven('../../data_mingjian/data/image_test/test.txt',
    #                                  '../../data_mingjian/data/image_test',
    #                                  transform=trans)
    # test_dataset = BoxPointFromSeven('/media/zjs/A22A53E82A53B7CD/kuaice/data_meidong/validation/dataset/real_data2_out_multiple.txt',
    #                                  '/media/zjs/A22A53E82A53B7CD/kuaice/data_meidong/validation/dataset/image',
    #                                  transform=trans)

    from datasets import BoxPointFromCPM
    test_dataset = BoxPointFromCPM('/media/zjs/A22A53E82A53B7CD/kuaice/data_meidong/validation/dataset/real_data2_out_multiple.txt',
                                   '/media/zjs/A22A53E82A53B7CD/kuaice/data_meidong/validation/dataset/all-real_data2_predict.txt',
                                   '/media/zjs/A22A53E82A53B7CD/kuaice/data_meidong/validation/dataset/image',
                                   transform=trans)

    # from datasets import BoxPointAsOne
    # test_dataset = BoxPointAsOne('../../data_mingjian/data/image_test/test.txt',
    #                              '../../data_mingjian/data/image_test',
    #                              transform=trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # # Test the model
    # basenet = models.resnet18(pretrained=False, num_classes=2)
    # # basenet.avgpool = nn.AdaptiveAvgPool2d(1)
    # basenet.avgpool = nn.AvgPool2d(3, stride=1)
    from networks import MobileNet
    basenet = MobileNet()

    checkpoint = torch.load('./checkpoints/mobilenet_new_0400.pth')
    basenet.load_state_dict(checkpoint)

    eval_point_regression(basenet, test_loader, batch_size, visualize=False)
