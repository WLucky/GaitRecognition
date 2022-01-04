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

from datasets.sampler import TripletSampler, InferenceSampler
from datasets.collate_fn import CollateFn
from datasets.dataset import DataSet
from datasets.transform import BaseSilCuttingTransform
from modeling.loss_aggregator import LossAggregator
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
            lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
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

    test_sampler = InferenceSampler(test_dataset, args.test_batch)
    '''
    batch_sampler: returns a batch of indices at a time
    collate_fn: merges a list of samples to form a mini-batch of Tensor(s)
    '''
    train_loader = tordata.DataLoader(
        dataset=train_dataset,
        batch_sampler=test_sampler,
        collate_fn=CollateFn(train_dataset.label_set, sample_type="all_ordered"),
        num_workers=1)

    test_loader = tordata.DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        collate_fn=CollateFn(test_dataset.label_set, sample_type="all_ordered"),
        num_workers=1)
    return train_loader, test_loader

def inputs_pretreament(inputs, training):
    """Conduct transforms on input data.

    Args:
        inputs: the input data.
    Returns:
        tuple: training data including inputs, labels, and some meta data.
    """
    seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
    seq_trfs = [BaseSilCuttingTransform()]

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

def save_ckpt(save_path, model, optimizer, scheduler, all_result, iteration):
    mkdir(osp.join(save_path, "checkpoints/"))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'all_result': all_result,
        'iteration': iteration}
    torch.save(checkpoint,
                osp.join(save_path, 'checkpoints/iter-{:0>5}.pt'.format(iteration)))
                
def data_visualization(save_path, all_result):
    mkdir(osp.join(save_path, "imgs/"))
    for type in ['nm', 'cl', 'bg']:
        key1 = "train_{}_acc".format(type)
        key2 = "test_{}_acc".format(type)

        plt.plot(all_result[key1], label=key1)
        plt.plot(all_result[key2], label=key2)
        plt.legend()
        plt.savefig(osp.join(save_path, 'imgs/{}_acc.png'.format(type)))
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
    if(args.fb):
        dir_format = '{args.model}_{flag}'

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

    result = {}

    return nm_acc2, bg_acc2, cl_acc2

def get_acc_each_angle(result_dict):
    NM_acc = result_dict["scalar/test_accuracy/NM"]
    BG_acc = result_dict["scalar/test_accuracy/BG"]
    CL_acc = result_dict["scalar/test_accuracy/CL"]

    return de_diag(NM_acc, True), de_diag(BG_acc, True), de_diag(CL_acc, True)
