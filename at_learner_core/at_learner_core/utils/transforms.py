import random
import collections
from PIL import Image, ImageFilter
import numpy as np
import cv2
import torch


class Transform4EachKey(object):
    """
    Apply all torchvision transforms to dict by each key
    """

    def __init__(self, transforms, key_list=['data']):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        for key in self.key_list:
            input_dict[key] = self.transforms(input_dict[key])
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key_list)
        format_string += '\n)'
        return format_string


class Transform4EachElement(object):
    """
    Apply all transforms to list for each element
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_list):
        for idx in range(len(input_list)):
            input_list[idx] = self.transforms(input_list[idx])
        return input_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += '\n)'
        return format_string


class StackTensors(object):
    """
    Stack list of tensors to one tensor
    """
    def __init__(self):
        pass
    def __call__(self, input_list):
        return torch.stack(input_list)
    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTranspose(object):
    """
    Apply random rotation for data [0, 90, 180, 270] and class for predict
    """

    def __init__(self, data_column, target_column):
        if isinstance(data_column, str):
            data_column = [data_column]
        self.data_column = data_column
        self.target_column = target_column
        self.rotations = [0, 90, 180, 270]

    def __call__(self, input_dict):
        random_index = np.random.randint(0, 4)
        for column in self.data_column:
            input_dict[column] = input_dict[column].rotate(random_index * 90)

        input_dict[self.target_column] = torch.Tensor([random_index]).long()
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.data_column) + ', '
        format_string += str(self.target_column) + ')'
        return format_string


class GaussianBlur(object):
    """
    Apply Gaussian blur to image with probability 0.5
    """

    def __init__(self, max_blur_kernel_radius=3, rand_prob=0.5):
        self.max_radius = max_blur_kernel_radius
        self.rand_prob = rand_prob

    def __call__(self, img):
        radius = np.random.uniform(0, self.max_radius)
        if np.random.random() < self.rand_prob:
            return img.filter(ImageFilter.GaussianBlur(radius))
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.max_radius)


class GaussianNoise(object):
    """
    Apply Gaussian noise to image with probability 0.5
    """
    def __init__(self, var_limit=(10.0, 50.0), mean=0., rand_prob=0.5):
        self.var_limit = var_limit
        self.mean = mean
        self.rand_prob = rand_prob

    def __call__(self, img):
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        np_img = np.array(img)
        gauss = np.random.normal(self.mean, sigma, np_img.shape)
        if np.random.random() < self.rand_prob:
            np_img = np_img.astype(np.float32) + gauss
            np_img = np.clip(np_img, 0.0, 255.)
            img = Image.fromarray(np_img.astype(np.uint8))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(var_limit={0}, mean={1}, rand_prob={2})'.format(self.var_limit,
                                                                                           self.mean,
                                                                                           self.rand_prob)


class ResizeOpencv(object):
    """
    Apply resize with opencv function
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, out_type='PIL'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.out_type = out_type

    def __call__(self, img):
        if type(img) != np.ndarray:
            img = np.array(img)
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        if self.out_type == 'PIL':
            img = Image.fromarray(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0},interpolation={1},out_type={2})'.format(self.size,
                                                                                            self.interpolation,
                                                                                            self.out_type)


class RandomBlur(object):
    """
    Apply random blur
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(radius=3)) if random.random() < self.p else x

    def __repr__(self):
        return self.__class__.__name__
