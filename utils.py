import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.utils.data as tordata
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os.path as osp
from tqdm import tqdm
import hashlib
import os
import matplotlib.pyplot as plt
from torchvision import transforms


from datasets.sampler import TripletSampler, InferenceSampler
from datasets.collate_fn import CollateFn
from datasets.dataset import DataSet
from datasets.transform import BaseSilCuttingTransform
from util_tools import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from util_tools import get_msg_mgr
from util_tools import Odict, mkdir
from util_tools import evaluation as eval_functions
from util_tools.evaluation import de_diag


msg_mgr = get_msg_mgr()

def get_optimizer(model, args):
    optimizer = optim.Adam
    optimizer = optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), 
            lr = args.lr, weight_decay = args.weight_decay)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[int(lr) for lr in args.decreasing_lr.split(',')], gamma = 0.1)
    return scheduler

def get_loader(args):
    train_dataset = DataSet(args.dataset_root, training = True, dataset_partition = args.dataset_partition, cache = False)
    test_dataset = DataSet(args.dataset_root, training = False, dataset_partition = args.dataset_partition, cache = False)

    train_sampler = TripletSampler(train_dataset, [int(b) for b in args.train_batch.split(',')])
    test_sampler = InferenceSampler(test_dataset, args.test_batch)
    '''
    batch_sampler: returns a batch of indices at a time
    collate_fn: merges a list of samples to form a mini-batch of Tensor(s)
    '''
    train_loader = tordata.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=CollateFn(train_dataset.label_set, sample_type="fixed_ordered"),
        num_workers=1)

    test_loader = tordata.DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        collate_fn=CollateFn(test_dataset.label_set, sample_type="all_ordered"),
        num_workers=1)
    return train_loader, test_loader

def get_loader_for_test(args):
    train_dataset = DataSet(args.dataset_root, training = True, dataset_partition = args.dataset_partition, cache = False)
    test_dataset = DataSet(args.dataset_root, training = False, dataset_partition = args.dataset_partition, cache = False)

    train_eval_sampler = InferenceSampler(train_dataset, args.test_batch)
    test_sampler = InferenceSampler(test_dataset, args.test_batch)
    '''
    batch_sampler: returns a batch of indices at a time
    collate_fn: merges a list of samples to form a mini-batch of Tensor(s)
    '''
    train_loader = tordata.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_eval_sampler,
        collate_fn=CollateFn(train_dataset.label_set, sample_type="all_ordered"),
        num_workers=1)

    test_loader = tordata.DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        collate_fn=CollateFn(test_dataset.label_set, sample_type="all_ordered"),
        num_workers=1)
    return train_loader, test_loader

def inputs_pretreament(inputs, training, random_crop = False):
    """Conduct transforms on input data.

    Args:
        inputs: the input data.
    Returns:
        tuple: training data including inputs, labels, and some meta data.
    """
    seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
    seq_trfs = [BaseSilCuttingTransform()]

    if training and random_crop:
        train_transform = transforms.Compose([
            transforms.RandomCrop((64, 44), padding=4),
        ])
        seq_trfs = [train_transform]

    requires_grad = bool(training)
    seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
            for trf, seq in zip(seq_trfs, seqs_batch)]

    typs = typs_batch
    vies = vies_batch

    labs = list2var(labs_batch).long()

    if seqL_batch is not None:
        seqL_batch = np2var(seqL_batch).int()
    seqL = seqL_batch

    if seqL is not None:
        seqL_sum = int(seqL.sum().data.cpu().numpy())
        ipts = [_[:, :seqL_sum] for _ in seqs]
    else:
        ipts = seqs
    del seqs
    return ipts, labs, typs, vies, seqL

def train_step(optimizer, scheduler, Scaler, loss_sum, enable_float16 = True) -> bool:
    """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

    Args:
        loss_sum:The loss of the current batch.
    Returns:
        bool: True if the training is finished, False otherwise.
    """

    optimizer.zero_grad()
    if loss_sum <= 1e-9:
        msg_mgr.log_warning(
            "Find the loss sum less than 1e-9 but the training process will continue!")

    if enable_float16:
        Scaler.scale(loss_sum).backward()
        Scaler.step(optimizer)
        scale = Scaler.get_scale()
        Scaler.update()
        # Warning caused by optimizer skip when NaN
        # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
        if scale != Scaler.get_scale():
            msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                scale, Scaler.get_scale()))
            return False
    else:
        loss_sum.backward()
        optimizer.step()

    scheduler.step()
    return True

def save_ckpt(save_path, flag, model, optimizer, scheduler, all_result, iteration):
    mkdir(osp.join(save_path, "checkpoints/"))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'all_result': all_result,
        'iteration': iteration}
    torch.save(checkpoint,
                osp.join(save_path, 'checkpoints/{}-iter{:0>5}.pt'.format(flag, iteration)))

def data_visualization(save_path, all_result):
    mkdir(osp.join(save_path, "imgs/"))
    for type in ['nm', 'cl', 'bg']:
        key1 = "train_{}_acc".format(type)
        key2 = "test_{}_acc".format(type)

        plt.plot(all_result['test_iterations'], all_result[key1], label=key1)
        plt.plot(all_result['test_iterations'], all_result[key2], label=key2)
        plt.legend()
        plt.savefig(osp.join(save_path, 'imgs/{}_acc.png'.format(type)))
        plt.close()

    for type in ['nm', 'cl', 'bg']:
        key2 = "test_{}_acc".format(type)
        plt.plot(all_result['test_iterations'], all_result[key2], label=key2)

    plt.legend()
    plt.savefig(osp.join(save_path, 'imgs/test_acc.png'))
    plt.close()

def data_visualization_swa(save_path, all_result):
    mkdir(osp.join(save_path, "imgs/"))
    for type in ['nm', 'cl', 'bg']:
        key1 = "train_{}_acc".format(type)
        key2 = "test_{}_acc".format(type)
        key3 = "swa_test_{}_acc".format(type)

        plt.plot(all_result['test_iterations'], all_result[key1], label=key1)
        plt.plot(all_result['test_iterations'], all_result[key2], label=key2)
        plt.plot(all_result['test_iterations'], all_result[key3], label=key3)

        plt.legend()
        plt.savefig(osp.join(save_path, 'imgs/swa_{}_acc.png'.format(type)))
        plt.close()


def inference(model, test_loader):
    """Inference all the test data.

    Args:
        rank: the rank of the current process.Transform
    Returns:
        Odict: contains the inference results.
    """
    total_size = len(test_loader)
    pbar = tqdm(total=total_size, desc='Transforming')
    batch_size = test_loader.batch_sampler.batch_size
    rest_size = total_size
    info_dict = Odict()
    for inputs in test_loader:
        ipts = inputs_pretreament(inputs, training=False)
        with autocast(enabled=False):
            retval = model.forward(ipts)
            inference_feat = retval['inference_feat']
            # for k, v in inference_feat.items():
            #     inference_feat[k] = v

            del retval
        for k, v in inference_feat.items():
            inference_feat[k] = ts2np(v)
        info_dict.append(inference_feat)
        rest_size -= batch_size
        if rest_size >= 0:
            update_size = batch_size
        else:
            update_size = total_size % batch_size
        pbar.update(update_size)
    pbar.close()
    for k, v in info_dict.items():
        v = np.concatenate(v)[:total_size]
        info_dict[k] = v
    return info_dict

def run_test(model, test_loader):
    """Accept the instance object(model) here, and then run the test loop."""

    with torch.no_grad():
        info_dict = inference(model, test_loader)
        label_list = test_loader.dataset.label_list
        types_list = test_loader.dataset.types_list
        views_list = test_loader.dataset.views_list

        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        eval_func = eval_functions.identification
        return eval_func(info_dict)

def get_save_path(args):
    dir = ""
    dir_format = '{args.model}_iter{args.total_iter}_wd{args.weight_decay}_{flag}'
    if args.random_crop:
        dir_format = "random_crop_" + dir_format

    dir = dir_format.format(args = args, flag = hashlib.md5(str(args).encode('utf-8')).hexdigest()[:4])
    save_path = os.path.join(args.save_dir, dir)
    return save_path

def get_acc_info(result_dict):
    NM_acc = result_dict["scalar/test_accuracy/NM"]
    BG_acc = result_dict["scalar/test_accuracy/BG"]
    CL_acc = result_dict["scalar/test_accuracy/CL"]
    # nm_acc = np.mean(NM_acc)
    # bg_acc = np.mean(BG_acc)
    # cl_acc = np.mean(CL_acc)

    ##### acc exclude identical view
    nm_acc2 = de_diag(NM_acc)
    bg_acc2 = de_diag(BG_acc)
    cl_acc2 = de_diag(CL_acc)

    return nm_acc2, bg_acc2, cl_acc2

def get_acc_each_angle(result_dict):
    NM_acc = result_dict["scalar/test_accuracy/NM"]
    BG_acc = result_dict["scalar/test_accuracy/BG"]
    CL_acc = result_dict["scalar/test_accuracy/CL"]

    return de_diag(NM_acc, True), de_diag(BG_acc, True), de_diag(CL_acc, True)


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))