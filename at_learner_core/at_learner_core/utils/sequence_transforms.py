import numpy as np


class RandomChoiceOne(object):
    def __init__(self, key_list=['path']):
        self.key_list = key_list

    def __call__(self, items_dict):
        for key in self.key_list:
            items_dict[key] = [np.random.choice(items_dict[key])]
        return items_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.key_list) + ')'
        return format_string


class ChoiceOne(object):
    def __init__(self, index, key_list=['path']):
        self.index = index
        self.key_list = key_list

    def __call__(self, items_dict):
        for key in self.key_list:
            items_dict[key] = [items_dict[key][self.index]]
        return items_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.index) + ', '
        format_string += str(self.key_list) + ')'
        return format_string


class ContinuesRandomCrop(object):
    def __init__(self, start_index, length, key_list=['path']):
        self.start_index = start_index
        self.length = length
        self.key_list = key_list

    def __call__(self, items_dict):
        if type(self.start_index) == float:
            for k in self.key_list:
                length = len(items_dict[k])
            end_index = min(length - self.length, int(length * self.start_index))
            start_index = np.random.randint(0, end_index)
        elif type(self.max_start_index) == int:
            start_index = np.random.randint(0, self.start_index+1)

        for key in self.key_list:
            items_dict[key] = items_dict[key][start_index: start_index + self.length]
        return items_dict


class SequenceRandomCrop(object):
    def __init__(self, min_length, key_list=['path']):
        self.min_length = min_length
        self.key_list = key_list

    def __call__(self, items_dict):
        length = len(items_dict[self.key_list[0]])
        if length <= self.min_length:
            return items_dict
        seq_length = np.random.randint(self.min_length, length+1)
        start_index = np.random.randint(0, length - seq_length+1)
        for key in self.key_list:
            items_dict[key] = items_dict[key][start_index: start_index + seq_length]
        return items_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.min_length) + ', '
        format_string += str(self.key_list) + ')'
        return format_string


class LinspaceTransform(object):
    def __init__(self, num_elements, endpoint=False, key_list=['path'], max_start_index=0):
        self.num_elements = num_elements
        self.endpoint = endpoint
        self.key_list = key_list
        self.max_start_index = max_start_index

    def __call__(self, item_dict):

        if type(self.max_start_index) == float:
            for k in self.key_list:
                length = len(item_dict[k])
            start_index = np.random.randint(0, int(self.max_start_index * length))
        elif type(self.max_start_index) == int:
            start_index = np.random.randint(0, self.max_start_index+1)
        for key in self.key_list:
            arr = np.array(item_dict[key])
            idxes = np.linspace(start_index,
                                len(arr)-1,
                                num=self.num_elements,
                                endpoint=not self.endpoint,
                                dtype=int)
            item_dict[key] = arr[idxes]
        return item_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.num_elements) + ', '
        format_string += str(self.endpoint) + ', '
        format_string += str(self.key_list) + ', '
        format_string += str(self.max_start_index) + ')'
        return format_string


class ShuffleTransform(object):
    def __init__(self, p=0.5, sync_shuffle=False, key_list=['data']):
        self.p = p
        self.sync = sync_shuffle
        self.key_list = key_list

    def __call__(self, item_dict):
        if self.sync:
            arr_length = len(item_dict[self.key_list[0]])
            idxes = np.random.choice(range(arr_length), arr_length, replace=False)
            for key in self.key_list:
                if isinstance(item_dict[key], list):
                    item_dict[key] = np.array(item_dict[key])
                item_dict[key] = item_dict[key][idxes]
        else:
            for key in self.key_list:
                item_dict[key] = np.random.permutation(item_dict[key])

        return item_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.p) + ', '
        format_string += str(self.sync) + ', '
        format_string += str(self.key_list) + ')'
        return format_string


class DuplicateElements(object):
    def __init__(self, n_elems, random_elems, key_list, target_key, target_value, change_label):
        '''

        :param n_elems: (tuple, int)
        :param random_elems:
        :param key_list:
        :param target_key:
        :param target_value:
        :param change_label: bool
        '''
        self.n_elems = n_elems
        self.random_elems = random_elems
        self.key_list = key_list
        self.target_key = target_key
        self.target_value = target_value
        self.change_label = change_label

    def __call__(self, item_dict):
        if item_dict[self.target_key] == self.target_value:
            arr_length = len(item_dict[self.key_list[0]])
            if isinstance(self.n_elems, tuple):
                n_elems = np.random.randint(self.n_elems[0], self.n_elems[1]+1)
            else:
                n_elems = self.n_elems
            if self.random_elems:
                arr_idxes = np.random.choice(range(arr_length), n_elems, replace=False)
            else:
                start_idx = np.random.randint(0, arr_length - n_elems)
                arr_idxes = np.arange(start_idx, start_idx + n_elems)

            needed_idxes = np.linspace(0, len(arr_idxes), num=arr_length, endpoint=False, dtype=int)
            idxes = arr_idxes[needed_idxes]

            for key in self.key_list:
                if isinstance(item_dict[key], list):
                    item_dict[key] = np.array(item_dict[key])
                item_dict[key] = item_dict[key][idxes]

            if self.change_label:
                item_dict[self.target_key] = 1 - self.target_value
        return item_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += str(self.n_elems) + ', '
        format_string += str(self.random_elems) + ', '
        format_string += str(self.key_list) + ', '
        format_string += str(self.target_key) + ', '
        format_string += str(self.target_value) + ', '
        format_string += str(self.change_label) + ')'
        return format_string
