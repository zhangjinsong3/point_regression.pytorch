import numpy as np
from PIL import Image
import os
import random
import cv2

# suppress warnings and then import caffe
import sys
os.environ['GLOG_minloglevel'] = '2'
sys.path.append(os.path.join('/home/zjs/projects/python/MyPytorch/torchsample', 'python'))


from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from utils import resize_keep_ratio, visualize_gt, crop_image


class BoxPointAsOne(Dataset):
    '''
    Make a box point dataset for point regression from 7 points label and image
    Stitch the 7 images as an 1x7 big image
    '''
    def __init__(self, txt_file: str, root_dir: str, image_size=(640, 480), crop_size=(96, 96), transform=None):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.image_info = open(txt_file, 'r').readlines()
        self.image_size = image_size
        self.crop_size = crop_size

        self.data = []
        self.labels = []
        for i in range(len(self.image_info)):
            img_path = os.path.join(self.root_dir, self.image_info[i].split(' ')[0].strip())
            img = cv2.imread(img_path)

            points_seven = self.image_info[i].strip().replace(',', '').split(' ')[1:]
            assert len(points_seven)//2 == 7, 'the ground truth should contain 7 points information'
            points = []
            for j in range(len(points_seven)//2):
                point = (int(points_seven[2 * j]), int(points_seven[2 * j + 1]))
                # np_point = np.array([int(point_str[0]) / img.shape[1], int(point_str[1]) / img.shape[0]]) # scale to [0,1]
                points.append(point)

            # resize image to 640x480, keep ratio
            h, w, _ = img.shape
            img, trans = resize_keep_ratio(img, self.image_size)
            for i in range(len(points)):
                resize_ratio = trans[0]
                pad_l = trans[1]
                pad_t = trans[2]
                points[i] = (int(points[i][0] * resize_ratio + pad_l), int(points[i][1] * resize_ratio + pad_t))

            # resize image to 640x480
            # h, w, _ = img.shape
            # img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)
            # ratio_w = self.image_size[0] / w
            # ratio_h = self.image_size[1] / h
            # for i in range(len(points)):
            #     points[i] = (int(points[i][0] * ratio_w), int(points[i][1] * ratio_h))

            # visualize
            # visualize_gt(img, points)

            np_points = np.array(points)
            self.data.append(np.array(img))
            self.labels.append(np_points)

    def __getitem__(self, item):
        img_index = item
        img, points = self.data[img_index], self.labels[img_index]

        # crop point contained image from 7 points image
        # pad source image to prevent from 越界
        pad_img = np.pad(img, (self.crop_size, self.crop_size, (0, 0)), mode='constant')
        points_tmp = []
        for point in points:
            points_tmp.append((point[0] + self.crop_size[0], point[1] + self.crop_size[1]))
        points = points_tmp
        # visualize_gt(pad_img, points)

        # random deviation from -24 to 24
        rand_deviation = np.random.uniform(-24, 24, size=(7, 2)).astype('int')
        np_points = np.array(points) + rand_deviation

        crop_imgs = []
        crop_points = []
        for i in range(len(points)):
            crop_img, pad = crop_image(pad_img, np_points.astype('int')[i], self.crop_size)
            crop_point = np.array(self.crop_size) // 2 - rand_deviation[i] + np.array(pad) + \
                         np.array((i * self.crop_size[0], 0))
            crop_imgs.append(crop_img)
            crop_points.append(crop_point)
        crop_img_stitch = cv2.hconcat(crop_imgs)
        # assert crop_point[0] in range(24, 73), 'gt 应该在24-73之间'
        # assert crop_point[1] in range(24, 73), 'gt 应该在24-73之间'

        # cv2.circle(crop_img_stitch, tuple(crop_points[2]), 3, (0, 0, 255), -1)
        # cv2.imshow('crop', crop_img_stitch)
        # cv2.waitKey()

        target = np.array(crop_points).reshape(-1).astype('float32')
        # crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        crop_img_stitch = Image.fromarray(crop_img_stitch)

        if self.transform is not None:
            crop_img_stitch = self.transform(crop_img_stitch)

        # test for transform
        # np_img = img.numpy() * 255
        # np_img = np_img.astype('uint8').transpose((1, 2, 0)).copy()
        # cv2.namedWindow('transform')
        # cv2.imshow('transform', np_img)
        # cv2.waitKey()

        return crop_img_stitch, target

    def __len__(self):
        return len(self.data)


class BoxPointFromCPM(Dataset):
    '''
    make a box point dataset for point regression from 7 points CPM output and ground truth
    '''
    def __init__(self, label_file: str, cpm_file: str, root_dir: str, image_size=(640, 480), crop_size=(96, 96), transform=None):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.image_info = open(label_file, 'r').readlines()
        self.cpm_info = open(cpm_file, 'r').readlines()
        self.image_size = image_size
        self.crop_size = crop_size

        self.data = []
        self.labels = []
        self.center = []
        for i in range(len(self.image_info)):
            img_path = os.path.join(self.root_dir, self.image_info[i].split(' ')[0].strip())
            img = cv2.imread(img_path)

            points_seven = self.image_info[i].strip().replace(',', '').split(' ')[1:]
            assert len(points_seven)//3 == 8, 'the ground truth should contain 8 points information'

            points_cpm = self.cpm_info[i].strip().replace(',', '').split(' ')[1:]
            assert len(points_cpm)//3 == 7, 'the cpm output should contain 7 points information'

            points_gt = []
            points_center = []
            for j in range(len(points_cpm)//3):
                point = (int(points_seven[3 * j]), int(points_seven[3 * j + 1]))
                # np_point = np.array([int(point_str[0]) / img.shape[1], int(point_str[1]) / img.shape[0]]) # scale to [0,1]
                points_gt.append(point)
                point_cpm = (int(float(points_cpm[3 * j])), int(float(points_cpm[3 * j + 1])))
                points_center.append(point_cpm)

            # visualize_gt(img, points_gt)

            # resize image to 640x480 keep ratio
            # h, w, _ = img.shape
            # img, trans = resize_keep_ratio(img, self.image_size)
            # for i in range(len(points_gt)):
            #     resize_ratio = trans[0]
            #     pad_l = trans[1]
            #     pad_t = trans[2]
            #     points_gt[i] = (round(points_gt[i][0] * resize_ratio + pad_l),
            #                     round(points_gt[i][1] * resize_ratio + pad_t))
            #     points_center[i] = (round(points_center[i][0] * resize_ratio + pad_l),
            #                         round(points_center[i][1] * resize_ratio + pad_t))


            # resize image to 640x480
            h, w, _ = img.shape
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.image_size[0] / w
            ratio_h = self.image_size[1] / h
            for i in range(len(points_gt)):
                points_gt[i] = (points_gt[i][0] * ratio_w, points_gt[i][1] * ratio_h)
                # points_center[i] = (int(points_center[i][0] * ratio_w), int(points_center[i][1] * ratio_h))

            # visualize
            # visualize_gt(img, points_center)

            np_points_gt = np.array(points_gt)
            np_points_center = np.array(points_center)
            self.data.append(np.array(img))
            self.labels.append(np_points_gt)
            self.center.append(np_points_center)

    def __getitem__(self, item):
        img_index = item//7
        point_index = item % 7
        img, points_gt, points_center = self.data[img_index], self.labels[img_index], self.center[img_index]

        # crop point contained image from 7 points image
        # pad source image to prevent from 越界
        pad_img = np.pad(img, (self.crop_size, self.crop_size, (0, 0)), mode='constant')
        points_tmp = []
        for point in points_gt:
            points_tmp.append((point[0] + self.crop_size[0], point[1] + self.crop_size[1]))
        points_gt = points_tmp

        points_center_tmp = []
        for point in points_center:
            if point[0] < 0 or point[1] < 0:
                points_center_tmp.append((-1, -1))
            else:
                points_center_tmp.append((point[0] + self.crop_size[0], point[1] + self.crop_size[1]))
        points_center = points_center_tmp
        # visualize_gt(pad_img, points)

        np_points_gt = np.array(points_gt)
        np_points_center = np.array(points_center).astype('int')
        if np_points_center[point_index][0] > 0 and np_points_center[point_index][1] > 0:
            crop_img, pad = crop_image(pad_img, np_points_center[point_index], self.crop_size)
            crop_point = np.array(self.crop_size) // 2 + np_points_gt[point_index] - np_points_center[point_index]
        else:
            crop_img = np.zeros((96, 96, 3), dtype='uint8')
            crop_point = np.array((-1, -1))

        # cv2.circle(crop_img, tuple(crop_point), 3, (0, 0, 255), -1)
        # cv2.imshow('crop', crop_img)
        # cv2.waitKey()

        target = crop_point.astype('float32')
        # crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        crop_img = Image.fromarray(crop_img)

        if self.transform is not None:
            crop_img = self.transform(crop_img)

        # test for transform
        # np_img = img.numpy() * 255
        # np_img = np_img.astype('uint8').transpose((1, 2, 0)).copy()
        # cv2.namedWindow('transform')
        # cv2.imshow('transform', np_img)
        # cv2.waitKey()

        return crop_img, target

    def __len__(self):
        return len(self.data) * 7


class BoxPointFromSeven(Dataset):
    '''
    make a box point dataset for point regression from 7 points label and image
    '''
    def __init__(self, txt_file: str, root_dir: str, image_size=(640, 480), crop_size=(96, 96), transform=None):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.image_info = open(txt_file, 'r').readlines()
        self.image_size = image_size
        self.crop_size = crop_size

        self.data = []
        self.labels = []
        for i in range(len(self.image_info)):
            img_path = os.path.join(self.root_dir, self.image_info[i].split(' ')[0].strip())
            img = cv2.imread(img_path)

            points_seven = self.image_info[i].strip().replace(',', '').split(' ')[1:]
            assert len(points_seven)//2 == 7, 'the ground truth should contain 7 points information'
            points = []
            for j in range(len(points_seven)//2):
                point = (int(points_seven[2 * j]), int(points_seven[2 * j + 1]))
                # np_point = np.array([int(point_str[0]) / img.shape[1], int(point_str[1]) / img.shape[0]]) # scale to [0,1]
                points.append(point)

            # resize image to 640x480 keep ratio
            h, w, _ = img.shape
            img, trans = resize_keep_ratio(img, self.image_size)
            for i in range(len(points)):
                resize_ratio = trans[0]
                pad_l = trans[1]
                pad_t = trans[2]
                points[i] = (int(points[i][0] * resize_ratio + pad_l), int(points[i][1] * resize_ratio + pad_t))

            # resize image to 640x480
            # h, w, _ = img.shape
            # img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)
            # ratio_w = self.image_size[0] / w
            # ratio_h = self.image_size[1] / h
            # for i in range(len(points)):
            #     points[i] = (int(points[i][0] * ratio_w), int(points[i][1] * ratio_h))

            # visualize
            # visualize_gt(img, points)

            np_points = np.array(points)
            self.data.append(np.array(img))
            self.labels.append(np_points)

    def __getitem__(self, item):
        # # 7 points together train one model
        # img_index = item//7
        # point_index = item % 7

        # each point trains a model
        img_index = item
        point_index = 0
        img, points = self.data[img_index], self.labels[img_index]

        # crop point contained image from 7 points image
        # pad source image to prevent from 越界
        pad_img = np.pad(img, (self.crop_size, self.crop_size, (0, 0)), mode='constant')
        points_tmp = []
        for point in points:
            points_tmp.append((point[0] + self.crop_size[0], point[1] + self.crop_size[1]))
        points = points_tmp
        # visualize_gt(pad_img, points)

        # random deviation from -24 to 24
        rand_deviation = np.random.uniform(-24, 24, size=(7, 2)).astype('int')
        np_points = np.array(points).astype('int') + rand_deviation

        crop_img, pad = crop_image(pad_img, np_points[point_index], self.crop_size)
        crop_point = np.array(self.crop_size) // 2 - rand_deviation[point_index] + np.array(pad)
        assert crop_point[0] in range(24, 73), 'gt 应该在24-73之间'
        assert crop_point[1] in range(24, 73), 'gt 应该在24-73之间'

        # cv2.circle(crop_img, tuple(crop_point), 3, (0, 0, 255), -1)
        # cv2.imshow('crop', crop_img)
        # cv2.waitKey()

        target = crop_point.astype('float32')
        # crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        crop_img = Image.fromarray(crop_img)

        if self.transform is not None:
            crop_img = self.transform(crop_img)

        # test for transform
        # np_img = img.numpy() * 255
        # np_img = np_img.astype('uint8').transpose((1, 2, 0)).copy()
        # cv2.namedWindow('transform')
        # cv2.imshow('transform', np_img)
        # cv2.waitKey()

        return crop_img, target

    def __len__(self):
        # return len(self.data) * 7 # 7 points together train one model
        return len(self.data) # each point trains a model


class BoxPoint(Dataset):
    """
    make a box point dataset for point regression
    """
    def __init__(self, txt_file: str, root_dir: str, ignore=[], transform=None):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.image_info = open(txt_file, 'r').readlines()

        self.data = []
        self.labels = []
        for i in range(len(self.image_info)):
            # 忽略以ignore 中元素开头的数据
            if self.image_info[i].split(' ')[0].strip()[0] in ignore:
                continue
            img_path = os.path.join(self.root_dir, self.image_info[i].split(' ')[0].strip())
            img = cv2.imread(img_path)
            self.data.append(np.array(img))
            point_str = self.image_info[i].split(' ')[1].strip().split(',')
            # np_point = np.array([int(point_str[0]) / img.shape[1], int(point_str[1]) / img.shape[0]]) # scale to [0,1]
            np_point = np.array([int(point_str[0]), int(point_str[1])])
            self.labels.append(np_point.astype('float32'))

    def __getitem__(self, item):
        img, target = self.data[item], self.labels[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # test for transform
        # np_img = img.numpy() * 255
        # np_img = np_img.astype('uint8').transpose((1, 2, 0)).copy()
        # cv2.namedWindow('transform')
        # cv2.imshow('transform', np_img)
        # cv2.waitKey()

        return img, target

    def __len__(self):
        return len(self.data)


class Plate2MNIST(Dataset):
    """
    make a MNIST-like dataset class
    """
    def __init__(self,  txt_file: str, root_dir: str, train=True, transform=None):
        super().__init__()
        self.train = train
        self.transform = transform
        self.root_dir = root_dir
        self.image_path = open(txt_file, 'r').readlines()

        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(self.image_path)):
                img_path = os.path.join(self.root_dir, self.image_path[i].split(' ')[0].strip())
                img = Image.open(img_path).resize((96, 32), Image.ANTIALIAS).convert('RGB')
                self.train_data.append(np.array(img))
                self.train_labels.append(int(self.image_path[i].split(' ')[-1].strip()))
            self.labels_set = set(self.train_labels)
        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(self.image_path)):
                img_path = os.path.join(self.root_dir, self.image_path[i].split(' ')[0].strip())
                img = Image.open(img_path).resize((96, 32), Image.ANTIALIAS).convert('RGB')
                self.test_data.append(np.array(img))
                self.test_labels.append(int(self.image_path[i].split(' ')[-1].strip()))
            self.labels_set = set(self.test_labels)

    def __getitem__(self, item):
        if self.train:
            img, target = self.train_data[item], self.train_labels[item]
        else:
            img, target = self.test_data[item], self.test_labels[item]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class TripletPlate(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __len__(self) -> int:
        return len(self.image_path)

    def __init__(self, txt_file: str, root_dir: str, mode: str, transform=None) -> None:
        super().__init__()
        self.image_path = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        if mode == 'test':
            self.train = 0

        else:
            self.train = 1
        self.data = []
        self.labels = []
        for i in range(len(self.image_path)):
            img_path = os.path.join(self.root_dir, self.image_path[i].split(' ')[0].strip())
            img = Image.open(img_path).resize((96, 32), Image.ANTIALIAS).convert('RGB')
            self.data.append(np.array(img))
            self.labels.append(int(self.image_path[i].split(' ')[-1].strip()))
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.data[index], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            img3 = self.data[negative_index]

            img1 = Image.fromarray(np.array(img1))
            img2 = Image.fromarray(img2)
            img3 = Image.fromarray(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            # return img1, img2, img3, label1, negative_label
            return (img1, img2, img3), []
        else:
            img1, label1 = self.data[index], self.labels[index]
            img1 = Image.fromarray(img1)
            if self.transform is not None:
                img1 = self.transform(img1)
            return img1, label1


class SequentialLoadingrate(Dataset):
    """
        make one sequential loadingrate data from folder
        """

    def __len__(self) -> int:
        return self.length

    def __init__(self, txt_file: str, root_dir: str, transform=None) -> None:
        super().__init__()
        self.label_path = txt_file
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        self.labels = []
        self.length = 0
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.paths.append(lines[i].split(' ')[0].strip())
                self.labels.append(int(float(lines[i].split(' ')[-1].strip())))
                self.length += 1
        f.close()

    def __getitem__(self, index):
        path, label = self.paths[index], self.labels[index]
        image_path = os.path.join(self.root_dir, path)
        img = Image.open(image_path).resize((224, 224), Image.ANTIALIAS).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def ASPN_image(image, NS):
    '''
    'ASPN' means add salt & pepper noise
    :param image: numpy.array, shape=[H,W,C]
    :param NS: noise strength (NS>0)
    :return: numpy.array, shape=[H,W,C]
    '''
    assert NS > 0
    ny_image = np.array(image, dtype=np.uint8)
    noise_mask = np.random.normal(0, 1,
                             size=(image.shape[0], image.shape[1]))
    for i in range(ny_image.shape[2]):
        ny_image[:,:,i] = (noise_mask >= NS) * 255 +\
                          (noise_mask <= (-NS)) * 0 +\
                          ((noise_mask > (-NS)) & (noise_mask < NS)) * ny_image[:,:,i]
    ny_image = ny_image.astype(np.uint8)
    return ny_image


class TripletLoadingrate(Dataset):
    """
    Train: For each sample creates from one loading or uploading process,
           正样本采用相邻帧或者同一帧增加噪声
           负样本选取间隔较远的帧
    Test:  待定
    """

    def __len__(self) -> int:
        return len(self.image_path)

    def __init__(self, txt_file: str, root_dir: str, transform=None) -> None:
        super().__init__()
        self.image_path = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index: int):
        img_name = self.image_path[index]
        img_path_anchor = os.path.join(self.root_dir, img_name.split(' ')[0].strip())
        img_path_positive = os.path.join(self.root_dir, img_name.split(' ')[1].strip())
        img_path_negative = os.path.join(self.root_dir, img_name.split(' ')[2].strip())
        label_anchor = int(img_name.split(' ')[-2])
        label_neg = int(img_name.split(' ')[-1])

        with Image.open(img_path_anchor) as img0:
            img0 = img0.resize((224, 224), Image.ANTIALIAS)
            image_anchor = img0.convert('RGB')
        with Image.open(img_path_positive) as img1:
            img1 = img1.resize((224, 224), Image.ANTIALIAS)
            img1 = Image.fromarray(ASPN_image(np.array(img1), 2))
            image_positive = img1.convert('RGB')
            # image_positive.show()
        with Image.open(img_path_negative) as img2:
            img2 = img2.resize((224, 224), Image.ANTIALIAS)
            image_negative = img2.convert('RGB')

        if self.transform is not None:
            image_anchor = self.transform(image_anchor)
            image_positive = self.transform(image_positive)
            image_negative = self.transform(image_negative)
        return image_anchor, image_positive, image_negative, abs(label_neg - label_anchor)


class TripletLoadingrate1(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __len__(self) -> int:
        return len(self.image_path)

    def __init__(self, txt_file: str, root_dir: str, mode: str, transform=None) -> None:
        super().__init__()
        self.image_path = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        if mode == 'test':
            self.train = 0
        else:
            self.train = 1
        self.data = []
        self.labels = []
        for i in range(len(self.image_path)):
            img_path = os.path.join(self.root_dir, self.image_path[i].split(' ')[0].strip())
            img = Image.open(img_path).resize((224, 224), Image.ANTIALIAS).convert('RGB')
            self.data.append(np.array(img))
            self.labels.append(int(self.image_path[i].split(' ')[-1].strip()))
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.data[index], self.labels[index]
            positive_index = index
            # while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
            list_pos = range(max(0, label1 - 20), min(100, label1 + 20))
            negative_label = np.random.choice(list(self.labels_set - set(list_pos)))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            img3 = self.data[negative_index]

            img1 = Image.fromarray(np.array(img1))
            img2 = Image.fromarray(img2)
            img3 = Image.fromarray(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, abs(self.labels[negative_index] - label1)
        else:
            img1, label1 = self.data[index], self.labels[index]
            img1 = Image.fromarray(img1)
            if self.transform is not None:
                img1 = self.transform(img1)
            return img1, label1


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        super().__init__()
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}
            # self.labels_set = set(self.train_labels.numpy())
            # self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
            #                          for label in self.labels_set}


        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            # self.labels_set = set(self.test_labels.numpy())
            # self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
            #                          for label in self.labels_set}
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
