
import os
import argparse
import torch
import torch.nn as nn
from modeling.models.gaitpart import gaitPart
from modeling.loss_aggregator import TripletLossAggregator
from util_tools import config_loader, init_seeds, params_count, get_msg_mgr
from utils import *


parser = argparse.ArgumentParser(description='Main program')
############################## data config ##############################
parser.add_argument('--cache', action='store_true', help="cache the dataset")
parser.add_argument('--dataset_root', type=str, help='location of the data corpus', required=True)
parser.add_argument('--dataset_partition', type=str, default="./partitions/partition.json", help='The path of partition config:trian set and test set')

############################## model config ##############################
parser.add_argument('--model', type=str, default='gaitpart', help="type of model")

############################## basic config ##############################
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--save_dir', type=str, default="result", help='The parent directory used to save the trained models')

############################## train config ################################
parser.add_argument('--total_iter', type=int, default=20000, help="total iteration to train")
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='10000,15000', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--test_iter', type=int, default=200, help="iter to test")
parser.add_argument('--train_batch', default='4,8', help='default: 4 label, 8 sample for each label')
parser.add_argument('--test_batch', default='16', help='test sample batch')

############################## log config ################################
parser.add_argument('--log_to_file', action='store_true', help="log to file")
parser.add_argument('--log_iter', type=int, default=100, help="iter to log")




if __name__ == '__main__':
    args = parser.parse_args()
    init_seeds(args.seed)
    save_path = get_save_path(args)
    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(save_path, args.log_to_file, args.log_iter, 0)
    msg_mgr.log_info(args)

    cfgs = config_loader(args.cfgs)

    model = gaitPart()
    model.cuda()
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")
    
    ########################## optimizer and scheduler ##########################
    loss_aggregator = TripletLossAggregator()
    Scaler = GradScaler()
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    train_loader, _ = get_loader(args)
    train_eval_loader, test_loader = get_loader_for_test(args)
    ########################## training process ##########################
    iteration = 0
    model.train()
    for inputs in train_loader:
        ipts = inputs_pretreament(inputs, training = True)
        with autocast(enabled=True):
            retval = model(ipts)
            training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
            del retval
        loss_sum, loss_info = loss_aggregator(training_feat)
        ok = train_step(optimizer, scheduler, Scaler, loss_sum)
        if ok:
            iteration += 1
        else:
            continue

        visual_summary.update(loss_info)
        visual_summary['scalar/learning_rate'] = optimizer.param_groups[0]['lr']

        msg_mgr.train_step(loss_info, visual_summary)

        ########################## testing process ##########################

        if iteration % args.test_iter == 0:
            # save the checkpoint
            # save_ckpt(save_path, model, optimizer, scheduler,  iteration, engine_cfg)
            msg_mgr.log_info("Running test...")
            model.eval()
            msg_mgr.log_info("Eval for train dataset...")
            result_dict = run_test(model, train_eval_loader)
            msg_mgr.log_info("Eval for test dataset...")
            result_dict = run_test(model, test_loader)
            model.train()
            msg_mgr.reset_time()
            
        if iteration >= args.total_iter:
            break
