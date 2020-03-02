import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == list:
            val, n = val
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMetricsMeter(object):
    def __init__(self, metric_config=None):
        self.loss = AverageMeter()
        self.metric = get_meter(metric_config.metric)
        self.target_column = metric_config.metric.target_column

    def reset(self):
        self.loss.reset()
        if self.metric:
            self.metric.reset()

    def update(self, loss, output_dict):
        self.loss.update(loss)
        if self.metric:
            # self.target_column and output as args and other info from output_dict
            self.metric.update(output_dict.pop(self.target_column),
                               output_dict.pop('output'),
                               output_dict)


class AccuracyMeter(object):
    def __init__(self, meter_config):
        self.nclasses = meter_config.nclasses
        self.target = np.empty((0), dtype=np.int32)
        self.predict = np.empty((0, self.nclasses), dtype=np.float32)

    def reset(self):
        self.target = np.empty((0), dtype=np.int32)
        self.predict = np.empty((0, self.nclasses), dtype=np.float32)

    def update(self, target, predict, other_info):
        self.target = np.hstack((self.target, target))
        self.predict = np.vstack((self.predict, predict))

    def get_accuracy(self, target_class=-1):
        if target_class == -1:
            predict = self.predict.argmax(1)

        accuracy = accuracy_score(self.target, predict)
        return accuracy


class ACERMeter(object):
    def __init__(self):
        self.target = np.ones(0)
        self.output = np.ones(0)
        self.other_info = {}

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)
        self.other_info = {}

    def update(self, target, output, other_info):
        # If we use cross-entropy
        if len(output.shape) > 1:
            if output.shape[1] > 1:
                output = output[:, 1]
            elif output.shape[1] == 1:
                output = output[:, 0]
        if len(target.shape) > 1:
            target = target[:, 0]

        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output])

        if len(other_info) > 0:
            for k, v in other_info.items():
                if k in self.other_info:
                    self.other_info[k].extend(v)
                else:
                    self.other_info[k] = v

    def get_all_metrics(self, thr=0.5):
        ":return ACER, APCER, BPCER"

        result_dict = {}
        y_pred_bin = self.output.copy()
        y_pred_bin[y_pred_bin < thr] = 0
        y_pred_bin[y_pred_bin >= thr] = 1

        tn, fp, fn, tp = confusion_matrix(self.target, y_pred_bin).ravel()
        # print(TP)
        apcer = fp / (tn + fp)
        bpcer = fn / (fn + tp)
        acer = (apcer + bpcer) / 2
        result_dict[thr] = {'acer': acer,
                            'apcer': apcer,
                            'bpcer': bpcer}

        return result_dict


class ROCMeter(object):
    """Compute TPR with fixed FPR"""

    def __init__(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def update(self, target, output, other_info):
        # If we use cross-entropy
        if len(output.shape) > 1:
            if output.shape[1] > 1:
                output = output[:, 1]
            elif output.shape[1] == 1:
                output = output[:, 0]
        if len(target.shape) > 1:
            target = target[:, 0]

        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output])

        if len(other_info) > 0:
            for k, v in other_info.items():
                if k in self.other_info:
                    self.other_info[k].extend(v)
                else:
                    self.other_info[k] = v

    def get_roc_curve(self):
        fpr, tpr, thr = roc_curve(self.target, self.output)
        return fpr, tpr, thr

    def get_tpr(self, fixed_fpr):
        fpr, tpr, thr = roc_curve(self.target, self.output)
        tpr_filtered = tpr[fpr <= fixed_fpr]
        if len(tpr_filtered) == 0:
            return 0.0
        return tpr_filtered[-1]

    def get_accuracy(self, thr=0.5):
        acc = accuracy_score(self.target,
                             self.output >= thr)
        return acc

    def get_top_hard_examples(self, top_n=10):
        diff_arr = np.abs(self.target - self.output)
        hard_indexes = np.argsort(diff_arr)[::-1]
        hard_indexes = hard_indexes[:top_n]
        return hard_indexes, self.target[hard_indexes], self.output[hard_indexes]


def get_meter(meter_config):
    if meter_config is None:
        return None
    elif meter_config.name == "accuracy":
        return AccuracyMeter(meter_config)
    elif meter_config.name == 'tpr@fpr':
        return ROCMeter()
    elif meter_config.name == 'roc-curve':
        return ROCMeter()
    elif meter_config.name == 'acer':
        return ACERMeter()
