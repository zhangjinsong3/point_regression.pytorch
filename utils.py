from itertools import combinations

import numpy as np
import cv2
import torch


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def crop_image(image, point, size=(96, 96)):
    """
    crop size image from padded image
    :param image: input image
    :param point: crop center
    :param size:  crop size
    :return: crop image pad_l pad_r
    """
    h, w, _ = image.shape
    assert point[0] in range(w), 'gt x 越界'
    assert point[1] in range(h), 'gt y 越界'
    xmin = max(0, point[0] - size[0]//2)
    ymin = max(0, point[1] - size[1]//2)
    xmax = min(w, point[0] + size[0]//2)
    ymax = min(h, point[1] + size[1]//2)
    pad_l = max(0, 0 - point[0] + size[0]//2)
    pad_r = max(0, point[0] + size[0]//2 - w)
    pad_t = max(0, 0 - point[1] + size[1]//2)
    pad_b = max(0, point[1] + size[1]//2 - h)
    crop_img = image[ymin:ymax, xmin:xmax]
    crop_img = np.pad(crop_img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')
    assert crop_img.shape[0] > 0, '空图片'
    # cv2.imshow('crop_img', crop_img)
    # cv2.waitKey()
    return crop_img, (pad_l, pad_t)


def visualize_gt(Image, points):
    """
    visualize 7 points of box corners image
    :param Image: image
    :param points: 7 points coordinate
    :return: None
    """
    drewImg = Image.copy()

    cv2.line(drewImg, points[0], points[1], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[0], points[2], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[0], points[3], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[1], points[6], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[1], points[4], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[2], points[4], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[2], points[5], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[3], points[5], (255, 0, 0), thickness=3)
    cv2.line(drewImg, points[3], points[6], (255, 0, 0), thickness=3)

    cv2.namedWindow('image', 0)
    cv2.imshow('image', drewImg)
    cv2.waitKey()


def resize_keep_ratio(image, size=(640, 480)):
    """
    Resize image with the aspect ratio remain unchanged
    You use the resize ratio and pad_l pad_r to transform point coordinate
    :param image: input image
    :param size: destination image size
    :return: output image, resize ratio, pad_l, pad_r
    """
    h, w, _ = image.shape
    if w/h >= size[0]/size[1]:
        w_transit = size[0]
        resize_ratio = w_transit/ w
        h_transit = int(h * resize_ratio)
    else:
        h_transit = size[1]
        resize_ratio = h_transit / h
        w_transit = int(w * resize_ratio)
    image = cv2.resize(image, (w_transit, h_transit), interpolation=cv2.INTER_CUBIC)
    # resize_ratio = w_transit/w
    pad_l = (size[0] - w_transit)//2
    pad_r = (size[0]-w_transit) - pad_l
    pad_t = (size[1] - h_transit)//2
    pad_b = (size[1] - h_transit) - pad_t

    image_pad = np.pad(image, ((pad_t, pad_b),
                               (pad_l, pad_r),
                               (0, 0)), mode='constant')
    return image_pad, (resize_ratio, pad_l, pad_t)


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
