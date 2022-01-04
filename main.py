
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
parser.add_argument('--decreasing_lr', default='10000', help='decreasing strategy')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--test_iter', type=int, default=1000, help="iter to test")
parser.add_argument('--train_batch', default='4,8', help='default: 4 label, 8 sample for each label')
parser.add_argument('--test_batch', type=int, default='16', help='test sample batch')

############################## log config ################################
parser.add_argument('--log_to_file', action='store_true', help="log to file")
parser.add_argument('--log_iter', type=int, default=100, help="iter to log")

########################## SWA setting ##########################
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=5000, metavar='N', help='SWA start iteration number')
parser.add_argument('--swa_c_iters', type=int, default=100, metavar='N', help='SWA model collection frequency/cycle length in iterations')




if __name__ == '__main__':
    args = parser.parse_args()
    init_seeds(args.seed)
    save_path = get_save_path(args)
    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(save_path, args.log_to_file, args.log_iter, 0)
    msg_mgr.log_info(args)

    model = gaitPart()
    model.cuda()
    msg_mgr.log_info(params_count(model))

    if args.swa:
        swa_model = gaitPart()
        swa_model.cuda()
        swa_n = 0  

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

    all_result = {}
    all_result['train_result'] = []
    all_result['test_result'] = []
    all_result['train_nm_acc'] = []
    all_result['train_bg_acc'] = []
    all_result['train_cl_acc'] = []
    all_result['test_nm_acc'] = []
    all_result['test_bg_acc'] = []
    all_result['test_cl_acc'] = []
    all_result['test_iterations'] = []

    if args.swa:
        all_result['swa_test_nm_acc'] = []
        all_result['swa_test_bg_acc'] = []
        all_result['swa_test_cl_acc'] = []
        all_result['swa_test_result'] = []

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

        if args.swa and iteration >= args.swa_start and (iteration - args.swa_start) % args.swa_c_iters == 0:
            # SWA
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1

        ########################## testing process ##########################
        if iteration % args.test_iter == 0:
            # save the checkpoint
            msg_mgr.log_info("Running test...")
            model.eval()
            msg_mgr.log_info("Eval for train dataset...")
            train_result_dict = run_test(model, train_eval_loader)
            msg_mgr.log_info("Eval for test dataset...")
            test_result_dict = run_test(model, test_loader)
            model.train()
            msg_mgr.reset_time()
            
            train_nm_acc, train_bg_acc, train_cl_acc = get_acc_info(train_result_dict)
            test_nm_acc, test_bg_acc, test_cl_acc = get_acc_info(test_result_dict)

            #### gap
            msg_mgr.log_info("Gap Info:\tNM: %.3f,\tBG: %.3f,\tCL: %.3f" 
                %(train_nm_acc - test_nm_acc, train_bg_acc - test_bg_acc, train_cl_acc - test_cl_acc))

            #### save data
            all_result['train_result'].append(train_result_dict)
            all_result['test_result'].append(test_result_dict)
            all_result['train_nm_acc'].append(train_nm_acc)
            all_result['train_bg_acc'].append(train_bg_acc)
            all_result['train_cl_acc'].append(train_cl_acc)
            all_result['test_nm_acc'].append(test_nm_acc)
            all_result['test_bg_acc'].append(test_bg_acc)
            all_result['test_cl_acc'].append(test_cl_acc)
            all_result['test_iterations'].append(iteration)

            ####swa...
            if args.swa and iteration >= args.swa_start:
                msg_mgr.log_info("Eval for swa...")
                swa_model.eval()
                swa_test_result_dict = run_test(swa_model, test_loader)
                swa_test_nm_acc, swa_test_bg_acc, swa_test_cl_acc = get_acc_info(swa_test_result_dict)

                all_result['swa_test_nm_acc'].append("swa_test_nm_acc")
                all_result['swa_test_bg_acc'].append("swa_test_bg_acc")
                all_result['swa_test_cl_acc'].append("swa_test_cl_acc")
                all_result['swa_test_result'].append("swa_test_result_dict")
            elif args.swa:
                all_result['swa_test_nm_acc'].append(test_nm_acc)
                all_result['swa_test_bg_acc'].append(test_bg_acc)
                all_result['swa_test_cl_acc'].append(test_cl_acc)
                all_result['swa_test_result'].append(test_result_dict)

            save_ckpt(save_path, "normal", model, optimizer, scheduler, all_result, iteration)

            #### drow img
            data_visualization(save_path, all_result)
            if args.swa:
                data_visualization_swa(save_path, all_result)

        if iteration >= args.total_iter:
            break
