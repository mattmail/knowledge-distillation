import numpy as np
from tools.log import add_log, dict_to_msg


class History():
    def __init__(self, keys, title=''):
        super(History, self).__init__()
        self.title = '%s' % title
        self.dict = dict()
        for k in keys:
            self.dict[k] = []

    def update(self, new_dict, need_item=False):
        for k, v in self.dict.items():
            if k in new_dict.keys():
                if need_item:
                    self.dict[k].append(new_dict[k].item())
                else:
                    self.dict[k].append(new_dict[k])
            else:
                self.dict[k].append(0)

    def average(self):
        avg = dict()
        for k, v in self.dict.items():
            avg[k] = np.mean(np.asarray(v), axis=0)
        return avg

    def tail(self, opt=None):
        t = dict()
        if opt is not None:
            for k, v in self.dict.items():
                t[k] = v[-1][opt]
        else:
            for k, v in self.dict.items():
                t[k] = v[-1]
        return t

    def log_avg(self, config, format='%.04f\t'):
        add_log(config, dict_to_msg(self.average(), self.title, format))


class History_branch():
    def __init__(self, keys, title=''):
        super(History_branch, self).__init__()
        self.history_source = History(keys, title+'_source')
        self.history_target = History(keys, title+'_target')

    def update(self, new_dict, need_item=False):
        self.history_source.update(new_dict[0], need_item)
        self.history_target.update(new_dict[1], need_item)

    def average(self):
        avg_source = self.history_source.average()
        avg_target = self.history_target.average()
        return [avg_source, avg_target]

    def tail(self, opt=None):
        return [self.history_source.tail(opt), self.history_target.tail(opt)]

    def log_avg(self, config, format=None):
        self.history_source.log_avg(config, format)
        self.history_target.log_avg(config, format)
