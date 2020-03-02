import random
import collections
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import torch

import scipy.sparse
from sklearn import svm
from .pyflow import pyflow

from torchvision.transforms import functional as F

class CreateNewItem(object):
    def __init__(self, transforms, key, new_key):
        self.transforms = transforms
        self.key = key
        self.new_key = new_key

    def __call__(self, input_dict):
        input_dict[self.new_key] = self.transforms(input_dict[self.key])
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key) + ', ' + str(self.new_key)
        format_string += ')'
        return format_string

class RandomZoom(object):
    def __init__(self, size):
        self.size_min = size[0]
        self.size_max = size[1]
                       
    def __call__(self, imgs):
        p_size = np.random.randint(self.size_min, self.size_max+1)
        size = (int(p_size), int(p_size))
        out = list(F.center_crop(img, size) for img in imgs)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + '(size={}-{})'.format(self.size_min,self.size_max)


class LiuOpticalFlowTransform(object):
    def __init__(self, first_index, second_index):
        self.first_index = first_index
        self.second_index = second_index
        
    def __call__(self, images):
        if type(self.first_index) == tuple:
            first_index = np.random.randint(self.first_index[0], self.first_index[1]+1)
        else:
            first_index = self.first_index

        if type(self.second_index) == tuple:
            second_index = np.random.randint(self.second_index[0], self.second_index[1])
        else:
            second_index = self.second_index
        
        
        im1 = images[first_index]
        im2 = images[second_index]
        im1 = np.array(im1).astype(float) / 255.
        im2 = np.array(im2).astype(float) / 255.
        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha=0.012, ratio=0.75, minWidth=20,
                                             nOuterFPIterations=7, nInnerFPIterations=1,
                                             nSORIterations=30, colType=0)
        return [u.astype(np.float32), v.astype(np.float32)]
    

class SaveOnlyMaxDiff(object):
    def __init__(self, first_index_range, second_index_range):
        self.first_index_range = first_index_range
        self.second_index_range = second_index_range

    def __call__(self, images):
        max_diff = 0
        max_first_index, max_second_index = None, None
        for first_index in self.first_index_range:
            first_np_arr = np.array(images[first_index])
            for second_index in self.second_index_range:
                diff = np.abs(first_np_arr - np.array(images[second_index])).sum()
                if diff > max_diff:
                    max_first_index = first_index
                    max_second_index = second_index
                    max_diff = diff

        return [images[max_first_index], images[max_second_index]]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.first_index_range) + ', '
        format_string += str(self.second_index_range) + ', '
        format_string += ')'
        return format_string
    

class OpticalFlowTransform(object):
    def __init__(self, first_index, second_index, flow_type='all', return_type='PIL',
                 pyr_scale=0.5, levels=3, winsize=15,
                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.flow_type = flow_type
        self.return_type = return_type
        self.first_index = first_index
        self.second_index = second_index

        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def __call__(self, images):
        if type(self.first_index) == tuple:
            first_index = np.random.randint(self.first_index[0], self.first_index[1]+1)
        else:
            first_index = self.first_index

        if type(self.second_index) == tuple:
            second_index = np.random.randint(self.second_index[0], self.second_index[1])
        else:
            second_index = self.second_index

        first_img = cv2.cvtColor(np.array(images[first_index]), cv2.COLOR_RGB2GRAY)
        second_img = cv2.cvtColor(np.array(images[second_index]), cv2.COLOR_RGB2GRAY)

        flows = cv2.calcOpticalFlowFarneback(first_img, second_img, None,
                                             self.pyr_scale, self.levels, self.winsize,
                                             self.iterations, self.poly_n, self.poly_sigma, self.flags)

        if self.return_type == 'PIL':
            if self.flow_type == 'm':
                flows_mag = Image.fromarray(
                    cv2.normalize(flows[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode='L')
                return flows_mag
            elif self.flow_type == 'a':
                flows_ang = Image.fromarray(
                    cv2.normalize(flows[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode='L')
                return flows_ang
            elif self.flow_type == 'all':
                flows_mag = Image.fromarray(
                    cv2.normalize(flows[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode='L')
                flows_ang = Image.fromarray(
                    cv2.normalize(flows[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode='L')
                return [flows_mag, flows_ang]
        else:
            if self.flow_type == 'm':
                return flows[..., 0]
            elif self.flow_type == 'a':
                return flows[..., 1]
            elif self.flow_type == 'all':
                return flows


class DeleteKeys(object):
    def __init__(self, key):
        if type(key) == str:
            self.key_list = [key]
        else:
            self.key_list = key

    def __call__(self, input_dict):
        for del_key in self.key_list:
            input_dict.pop(del_key)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.key_list)
        format_string += ')'
        return format_string


class ApplyJoinTransforms2Dict(object):
    def __init__(self, transforms, key_list):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        input_list = [input_dict[x] for x in self.key_list]
        for t in self.transforms:
            input_list = t(input_list)

        for idx, key in enumerate(self.key_list):
            input_dict[key] = input_list[idx]
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.key_list)
        format_string += ')'
        return format_string


class Transform4EachLabel(object):
    """
    Applies transforms only to chosen labels
    """
    
    def __init__(self, transforms, label='target', allowed_labels=[0, 1]):
        self.label = label
        self.allowed_labels = allowed_labels if type(allowed_labels) == list else [allowed_labels]
        self.transforms = transforms
        
    def __call__(self, input_dict):
        dict_label = input_dict[self.label]
        if dict_label in set(self.allowed_labels):
            return self.transforms(input_dict)
        else:
            return input_dict
    
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += ',\n'
        format_string += str(self.label)
        format_string += '\n)'
        return format_string
    

class Transform4EachKey(object):
    """
    Apply all torchvision transforms to dict by each key
    """

    def __init__(self, transforms, key_list=['data']):
        self.transforms = transforms
        self.key_list = key_list

    def __call__(self, input_dict):
        for key in self.key_list:
            for t in self.transforms:
                input_dict[key] = t(input_dict[key])
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
            for t in self.transforms:
                input_list[idx] = t(input_list[idx])
        return input_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += self.transforms.__repr__()
        format_string += '\n)'
        return format_string


class JointTransform(object):
    """
    Apply all transforms with equal random parameters to each element
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    
    def __call__(self, input):
        for t in self.tranforms:
            input = t(input)
        return input
    
    
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
    def __init__(self, squeeze=False):
        self.squeeze = squeeze

    def __call__(self, input_list):
        res_tensor = torch.stack(input_list)
        if self.squeeze:
            res_tensor = res_tensor.squeeze()
        return res_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'({self.squeeze})'


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
    
    
    
class SquarePad(object):
    
    def __call__(self, im):
        if type(im)==list:
            return [self.__call__(ims) for ims in im]
        w,h = im.size
        max_size = max(w,h)

        delta_w = max_size - w
        delta_h = max_size - h

        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(im, padding)
        return new_im
    
    def __repr__(self):
        return self.__class__.__name__
    
    
class RemoveBlackBorders(object):
    
    def __call__(self, im):
        if type(im)==list:
            return [self.__call__(ims) for ims in im]
        V = np.array(im)
        V = np.mean(V,axis=2)
        X=np.sum(V,axis=0)
        Y=np.sum(V,axis=1)
        y1=np.nonzero(Y)[0][0]
        y2=np.nonzero(Y)[0][-1]

        x1=np.nonzero(X)[0][0]
        x2=np.nonzero(X)[0][-1]
        return im.crop([x1,y1,x2,y2])
    
    def __repr__(self):
        return self.__class__.__name__


class MeanSubtraction(object):
    def __call__(self, images):
        mean_image = np.zeros(np.array(images[0]).shape)
        for i in images:
            mean_image = mean_image + np.array(i).astype(np.float32)
        mean_image = mean_image / len(images)
        diff_images = []
        for i in images:
            diff_image = np.array(i) - mean_image
            diff_images.append(Image.fromarray(np.abs(diff_image).astype(np.uint8)))
        return diff_images

    def __repr__(self):
        return self.__class__.__name__


    
    
    
class MeanXSubtraction(object):
    def __init__(self, x):
        self.x = x

    def __call__(self, images):
        mean_image = np.zeros(np.array(images[0]).shape)
        for i in images:
            mean_image = mean_image + np.array(i).astype(np.float32)
        mean_image = mean_image / len(images)
        diff_images = np.zeros((len(images), mean_image.shape[0], mean_image.shape[1], mean_image.shape[2]))
        for idx, img in enumerate(images):
            diff_image = np.array(img) - mean_image
            diff_images[idx] = np.abs(diff_image)

        if self.x == 'min':
            result_arr = diff_images.min(axis=0)
        elif self.x == 'max':
            result_arr = diff_images.max(axis=0)
        elif self.x == 'mean':
            result_arr = diff_images.mean(axis=0)

        result_image = Image.fromarray(result_arr.astype(np.uint8))
        return result_image

    def __repr__(self):
        return self.__class__.__name__

    
class SelectOneImg(object):
    def __init__(self, n):
        self.number = n
    def __call__(self, images):
        return images[self.number]
    
    def __repr__(self):
        return self.__class__.__name__    
    

class MergeTransform(object):
    def __init__(self, key_list, save_key):
        self.key_list = key_list
        self.save_key = save_key

    def __call__(self, input_dict):
        result_list = []
        for key in self.key_list:
            result_list.append(input_dict[key])

        input_dict[self.save_key] = result_list
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.key_list) + ', ' + str(self.save_key)
        format_string += ')'
        return format_string


class RankPooling(object):
    def __init__(self, C, nonlinear='ssr'):
        self.C = C
        self.nonlinear = nonlinear

    def _smoothSeq(self, seq):
        res = np.cumsum(seq, axis=1)
        seq_len = np.size(res, 1)
        res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
        return res

    def _rootExpandKernelMap(self, data):

        element_sign = np.sign(data)
        nonlinear_value = np.sqrt(np.fabs(data))
        return np.vstack((nonlinear_value * (element_sign > 0), nonlinear_value * (element_sign < 0)))

    def _getNonLinearity(self, data, nonLin='ref'):

        # we don't provide the Chi2 kernel in our code
        if nonLin == 'none':
            return data
        if nonLin == 'ref':
            return self._rootExpandKernelMap(data)
        elif nonLin == 'tanh':
            return np.tanh(data)
        elif nonLin == 'ssr':
            return np.sign(data) * np.sqrt(np.fabs(data))
        else:
            raise ("We don't provide {} non-linear transformation".format(nonLin))

    def _normalize(self, seq, norm='l2'):

        if norm == 'l2':
            seq_norm = np.linalg.norm(seq, ord=2, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        elif norm == 'l1':
            seq_norm = np.linalg.norm(seq, ord=1, axis=0)
            seq_norm[seq_norm == 0] = 1
            seq_norm = seq / np.expand_dims(seq_norm, 0)
            return seq_norm
        else:
            raise ("We only provide l1 and l2 normalization methods")

    def _rank_pooling(self, time_seq, NLStyle='ssr'):
        '''
        This function only calculate the positive direction of rank pooling.
        :param time_seq: D x T
        :param C: hyperparameter
        :param NLStyle: Nonlinear transformation.Including: 'ref', 'tanh', 'ssr'.
        :return: Result of rank pooling
        '''

        seq_smooth = self._smoothSeq(time_seq)
        seq_nonlinear = self._getNonLinearity(seq_smooth, NLStyle)
        seq_norm = self._normalize(seq_nonlinear)
        seq_len = np.size(seq_norm, 1)
        Labels = np.array(range(1, seq_len + 1))
        seq_svr = scipy.sparse.csr_matrix(np.transpose(seq_norm))
        svr_model = svm.LinearSVR(epsilon=0.1,
                                  tol=0.001,
                                  C=self.C,
                                  loss='squared_epsilon_insensitive',
                                  fit_intercept=False,
                                  dual=False,
                                  random_state=42)
        svr_model.fit(seq_svr, Labels)
        return svr_model.coef_

    def __call__(self, images):
        np_images = np.array([np.array(x) for x in images])
        input_arr = np_images.reshape((np_images.shape[0], -1)).T
        result_img = self._rank_pooling(input_arr).reshape(np_images.shape[1:])
        result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
        return Image.fromarray((result_img * 255).astype(np.uint8))

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(C={self.C}, '
        format_string += f'nonlinear={self.nonlinear})'
        return format_string