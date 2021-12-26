import os
import os.path as osp
import torch.utils.data as tordata
from PIL import Image


class DataSet(tordata.Dataset):
    def __init__(self, dataset_root, cache, transform):
        self.dataset_root = dataset_root
        self.__dataset_parser()
        self.cache = cache
        self.transform = transform
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        # label map to the index in seq_info
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)

        if self.cache:
            self.__load_all_data()

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.png'):
                image = Image.open(pth)  # 读取到的是RGB， W, H, C
                image = self.transform(image)   # transform转化image为：C, H, W
            else:
                raise ValueError('- Loader - just support .png !!!')
            data_list.append(image)
        for data in data_list:
            if len(data) != len(data_list[0]):
                raise AssertionError

        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, training):
        train_set = os.listdir(osp.join(self.dataset_root, "train"))
        # test_set = os.listdir(osp.join(self.dataset_root, "test"))
        train_set = [label for label in train_set if label.isdigit()]
        # test_set = [label for label in test_set if label.isdigit()]

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(self.dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(self.dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(self.dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            print('Find no file in %s-%s-%s.'%(lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(train_set)


