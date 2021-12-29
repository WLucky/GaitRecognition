import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.utils.data as tordata
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os.path as osp
from tqdm import tqdm
import pdb


import datasets.sampler as Samplers
from datasets.collate_fn import CollateFn
from datasets.dataset import DataSet
from datasets.transform import get_transform
from modeling.loss_aggregator import LossAggregator
from util_tools import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from util_tools import get_msg_mgr
from util_tools import Odict, mkdir, ddp_all_gather
from util_tools import evaluation as eval_functions


msg_mgr = get_msg_mgr()

def get_optimizer(model, optimizer_cfg):
    msg_mgr.log_info(optimizer_cfg)
    optimizer = get_attr_from([optim], optimizer_cfg['solver'])
    valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
    optimizer = optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), **valid_arg)
    return optimizer

def get_scheduler(optimizer, scheduler_cfg):
    msg_mgr.log_info(scheduler_cfg)
    Scheduler = get_attr_from(
        [optim.lr_scheduler], scheduler_cfg['scheduler'])
    valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
    scheduler = Scheduler(optimizer, **valid_arg)
    return scheduler

def get_loader(cfgs, train=True):
    data_cfg = cfgs['data_cfg']
    sampler_cfg = cfgs['trainer_cfg']['sampler'] if train else cfgs['evaluator_cfg']['sampler']
    dataset = DataSet(data_cfg, train)

    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        'sample_type', 'type'])
    sampler = Sampler(dataset, **vaild_args)

    '''
    batch_sampler: returns a batch of indices at a time
    collate_fn: merges a list of samples to form a mini-batch of Tensor(s)
    '''
    loader = tordata.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=data_cfg['num_workers'])
    return loader

def inputs_pretreament(inputs, training, cfgs):
    """Conduct transforms on input data.

    Args:
        inputs: the input data.
    Returns:
        tuple: training data including inputs, labels, and some meta data.
    """
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']

    seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
    trf_cfgs = engine_cfg['transform']
    seq_trfs = get_transform(trf_cfgs)

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

def train_step(optimizer, scheduler, Scaler, loss_sum, engine_cfg) -> bool:
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

    if engine_cfg['enable_float16']:
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

def save_ckpt(save_path, model, optimizer, scheduler,  iteration, engine_cfg):
    mkdir(osp.join(save_path, "checkpoints/"))
    save_name = engine_cfg['save_name']
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iteration': iteration}
    torch.save(checkpoint,
                osp.join(save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

def inference(model, test_loader, cfgs):
    """Inference all the test data.

    Args:
        rank: the rank of the current process.Transform
    Returns:
        Odict: contains the inference results.
    """
    training = False
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']

    total_size = len(test_loader)
    pbar = tqdm(total=total_size, desc='Transforming')
    batch_size = test_loader.batch_sampler.batch_size
    rest_size = total_size
    info_dict = Odict()
    for inputs in test_loader:
        ipts = inputs_pretreament(inputs, training, cfgs)
        with autocast(enabled=engine_cfg['enable_float16']):
            retval = model.forward(ipts)
            inference_feat = retval['inference_feat']
            for k, v in inference_feat.items():
                # inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                inference_feat[k] = v

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

def run_test(model, cfgs):
    """Accept the instance object(model) here, and then run the test loop."""
    test_loader = get_loader(cfgs, train = False)

    with torch.no_grad():
        info_dict = inference(model, test_loader, cfgs)
        label_list = test_loader.dataset.label_list
        types_list = test_loader.dataset.types_list
        views_list = test_loader.dataset.views_list

        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        if 'eval_func' in cfgs["evaluator_cfg"].keys():
            eval_func = cfgs['evaluator_cfg']["eval_func"]
        else:
            eval_func = 'identification'
        eval_func = getattr(eval_functions, eval_func)
        valid_args = get_valid_args(
            eval_func, model.cfgs["evaluator_cfg"], ['metric'])
        try:
            dataset_name = model.cfgs['data_cfg']['test_dataset_name']
        except:
            dataset_name = model.cfgs['data_cfg']['dataset_name']
        return eval_func(info_dict, dataset_name, **valid_args)

def run_train(model, cfgs, training = True):
    """Accept the instance object(model) here, and then run the train loop."""
    # 如果要多epoch 这些变量注意挪到函数外面  防止每次epoch都初始化
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    loss_aggregator = LossAggregator(cfgs['loss_cfg'])
    Scaler = GradScaler()
    optimizer = get_optimizer(model, cfgs['optimizer_cfg'])
    # pdb.set_trace()
    scheduler = get_scheduler(optimizer, cfgs['scheduler_cfg'])

    save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], engine_cfg['save_name'])

    dataloader = get_loader(cfgs, training)

    iteration = 0
    for inputs in dataloader:
        ipts = inputs_pretreament(inputs, training, cfgs)
        with autocast(enabled=engine_cfg['enable_float16']):
            retval = model(ipts)
            training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
            del retval
        loss_sum, loss_info = loss_aggregator(training_feat)
        ok = train_step(optimizer, scheduler, Scaler, loss_sum, engine_cfg)
        if ok:
            iteration += 1
        else:
            continue

        visual_summary.update(loss_info)
        visual_summary['scalar/learning_rate'] = optimizer.param_groups[0]['lr']

        msg_mgr.train_step(loss_info, visual_summary)

        if iteration % engine_cfg['save_iter'] == 0:
            # save the checkpoint
            save_ckpt(save_path, model, optimizer, scheduler,  iteration, engine_cfg)

            # run test if with_test = true
            if engine_cfg['with_test']:
                msg_mgr.log_info("Running test...")
                model.eval()
                result_dict = run_test(model, cfgs)
                model.train()
                msg_mgr.write_to_tensorboard(result_dict)
                msg_mgr.reset_time()
        if iteration >= engine_cfg['total_iter']:
            break