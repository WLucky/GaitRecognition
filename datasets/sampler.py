import math
import torch
import torch.utils.data as tordata

# Triplet  三次随机
# batchsize 8 16 : 从8个label 每个选8个例子索引
class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

    def __iter__(self):
        while True:
            sample_indices = []
            # pid --> label
            pid_list = random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                # 具有相同label的 数据(粒度：不同view) 的索引
                indices = self.dataset.indices_dict[pid]
                indices = random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            # 第三次只是打乱顺序
            if self.batch_shuffle:
                sample_indices = random_sample_list(
                    sample_indices, len(sample_indices))

            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def random_sample_list(obj_list, k):
    # Returns a random permutation of integers from 0 to n - 1.
    idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    idx = idx.tolist()
    return [obj_list[i] for i in idx]

# inference: 按照顺序返回
class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        if batch_size != 1:
            # dataset 按照batch_size的倍数向上取整
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        indx_batch = []

        for i in range(int(self.size / batch_size)):
            indx_batch.append(
                indices[i*batch_size: (i+1)*batch_size])

    def __iter__(self):
        yield from self.indx_batch

    def __len__(self):
        return len(self.dataset)
