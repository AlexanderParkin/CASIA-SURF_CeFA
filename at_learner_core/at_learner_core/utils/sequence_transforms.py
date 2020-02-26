import numpy as np


class LinspaceTransform(object):
    def __init__(self, num_elements, endpoint=False, key_list=['path']):
        self.num_elements = num_elements
        self.endpoint = endpoint
        self.key_list = key_list

    def __call__(self, item_dict):
        for key in self.key_list:
            arr = np.array(item_dict[key])
            idxes = np.linspace(0,
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
        format_string += str(self.key_list) + ')'
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