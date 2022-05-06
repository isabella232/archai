# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Full training, evaluation and metrics for NLP-based models.
"""

import argparse
import copy
import functools
import itertools
import logging
import math
import os
import pprint
import shutil
import sys
import time
from datetime import datetime
from typing import Tuple

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from archai.common import ml_perf_utils, utils
from archai.nlp.models.model_mixed_qat import MixedQATModel
from archai.nlp.compression.quantization.ptq import dynamic_quantization_torch_from_model
from archai.nlp.models.model_loader import (load_model_from_config,
                                           load_model_from_checkpoint)
from archai.nlp.compression.quantization.qat import (prepare_with_qat,
                                                     qat_to_float_modules)
from archai.nlp.datasets import exp_utils
from archai.nlp.datasets.distributed_utils import distributed as nv_distributed
from archai.nlp.datasets.distributed_utils.data_parallel import BalancedDataParallel
from archai.nlp.datasets.distributed_utils.data_utils import get_lm_corpus
from archai.nlp.datasets.exp_utils import (AverageMeter, create_exp_dir, l2_promote,
                                           log_env_info)
from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.models.model_utils import lamb_optimizer
from archai.nlp.models.model_utils.cyclic_cosine_scheduler import CyclicCosineDecayLR
from torch.nn.parallel import DistributedDataParallel

from archai.common.common import create_conf, create_dirs
from archai.common.config import Config


def parse_args(new_config):
    # parent_parser = argparse.ArgumentParser(
    #     description='PyTorch Transformer-XL Language Model',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #     add_help=False,
    #     )

    # parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    # cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    # # if debugging from VS then use toy mode otherwise use 1 GPU/FP32 mode to be on same side
    # default_config = 'dgx1_1gpu_fp32'
    # if utils.is_debugging():
    #     default_config = 'toy'

    # cfg_parser.add_argument('--config', default=default_config) # use 'dgx1_8gpu_fp16' for V100 16GB, dgx1_1gpu_fp16, default
    # cfg_parser.add_argument('--config_file', default='wt103_base.yaml')

    # config_args, _ = cfg_parser.parse_known_args()

    # if config_args.config is not None and config_args.config_file is not None:
    #     config_file_path = utils.full_path(os.path.join('.', 'archai', 'nlp', config_args.config_file))
    #     with open(config_file_path) as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']
    # else:
    #     config = {}

    # general = parser.add_argument_group('general setup')
    # general.add_argument('--work_dir', default='~/logdir', type=str,
    #                      help='Directory for the results')
    # general.add_argument('--experiment_name', default='mem_transformer', type=str,
    #                      help='Directory for the results')
    # general.add_argument('--append_dataset', action='store_true',
    #                      help='Automatically append dataset name to work_dir')
    # general.add_argument('--append_time', action='store_true',
    #                      help='Automatically append current time to work_dir')
    # general.add_argument('--cuda', action='store_true',
    #                      help='Run training on a GPU using CUDA')
    # general.add_argument('--fp16', action='store_true',
    #                      help='Run training in fp16/mixed precision')
    # general.add_argument('--restart', type=str, default='',
    #                      help='Restart training from the saved checkpoint')
    # general.add_argument('--pretrained_path', type=str, default='',
    #                      help='Absolute or relative pretrained model path for finetuning or QAT')
    # general.add_argument('--debug', action='store_true', default=None,
    #                      help='Run in debug mode (do not create exp dir)')
    # general.add_argument('--log_all_ranks', action='store_true',
    #                      help='Enable logging from all distributed ranks')
    # general.add_argument('--dllog_file', type=str, default='train_log.json',
    #                      help='Name of the DLLogger output file')
    # general.add_argument('--txtlog_file', type=str, default='train_log.log',
    #                      help='Name of the txt log file')
    # general.add_argument('--save_all', action='store_true',
    #                      help='Save all checkpoints')
    # general.add_argument('--no_env', action='store_false',
    #                      help='Do not print info on execution env')
    # general.add_argument('--no_train', action='store_false', default=False,
    #                      help='Only generate dataset caches, no training. Can be run on without GPU.')
    # general.add_argument('--no_eval', action='store_true',
    #                      help='Disable model evaluation')
    # general.add_argument('--refresh_cache', action='store_false', default=False,
    #                      help='Ignores any existing cache and overwrites it with new one')
    # general.add_argument('--log_interval', type=int, default=10,
    #                      help='Report interval')
    # general.add_argument('--target_throughput', type=float, default=None,
    #                      help='Target training throughput (for benchmarking)')
    # general.add_argument('--target_perplexity', type=float, default=None,
    #                      help='Target validation perplexity (for benchmarking)')
    # general.add_argument('--apex_amp_opt_level', type=str, default='O2',
    #                      choices=['O0', 'O1', 'O2', 'O3'],
    #                      help='Optimization level for apex amp')
    # general.add_argument('--affinity', type=str,
    #                      default='socket_unique_interleaved',
    #                      choices=['socket', 'single', 'single_unique',
    #                               'socket_unique_interleaved',
    #                               'socket_unique_continuous',
    #                               'disabled'],
    #                      help='type of CPU affinity')

    # dataset = parser.add_argument_group('dataset setup')
    # dataset.add_argument('--data', type=str, default=None,
    #                      help='Location of the data corpus')
    # general.add_argument('--cache_dir', default='cache', type=str,
    #                      help='Directory to store dataset cache, either absolute or relative')
    # dataset.add_argument('--dataset', type=str, # set to 'wt103' through config name unless toy mode when its wt2
    #                      choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8', 'olx_WordData20210110', 'olx_OutlookData20210917x2', 'olx_WordData20211003', 'olx_WordData20220118_S36', 'olx_RedditWA_S100'],
    #                      help='Dataset name')
    # dataset.add_argument('--vocab', type=str, default='word', choices=['word', 'bbpe', 'gpt2'],
    #                      help='Type of vocabulary')
    # dataset.add_argument('--vocab_size', type=int, default=None,
    #                      help='Size of vocabulary')

    # model = parser.add_argument_group('model setup - defaults are for base model')
    # model.add_argument('--model_type', default='mem_transformer', type=str,
    #                  choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
    #                  help='Which model type to use')
    # model.add_argument('--n_layer', type=int, default=16,
    #                    help='Number of total layers')
    # model.add_argument('--n_head', nargs='+', type=int, default=8,
    #                    help='Number of heads')
    # model.add_argument('--d_head', type=int, default=-1, # will be set by d_model // n_head
    #                    help='Head dimension')
    # model.add_argument('--d_embed', type=int, default=-1, # will be set from d_model
    #                    help='Embedding dimension')
    # model.add_argument('--d_model', type=int, default=512,
    #                    help='Model dimension')
    # model.add_argument('--d_inner', nargs='+', type=int, default=2048,
    #                    help='Inner dimension in feedforward layer')
    # model.add_argument('--dropout', type=float, default=0.1,
    #                    help='Global dropout rate')
    # model.add_argument('--dropatt', type=float, default=0.0,
    #                    help='Attention probability dropout rate')
    # model.add_argument('--pre_lnorm', action='store_true',
    #                    help='Apply LayerNorm to the input instead of the output')
    # model.add_argument('--attn_type', type=int, default=0,
    #                    help='Attention type. 0 for ours, 1 for Shaw et al,'
    #                    '2 for Vaswani et al, 3 for Al Rfou et al.')
    # model.add_argument('--not_tied', action='store_true',
    #                    help='Do not tie the word embedding and softmax weights')
    # model.add_argument('--clamp_len', type=int, default=-1,
    #                    help='Use the same pos embeddings after clamp_len')
    # model.add_argument('--adaptive', action='store_true',
    #                    help='Use adaptive softmax')
    # model.add_argument('--div_val', type=int, default=1,
    #                    help='Dividend value for adaptive input and softmax')
    # model.add_argument('--sample_softmax', type=int, default=-1,
    #                    help='Number of samples in sampled softmax')
    # model.add_argument('--init', default='normal', type=str,
    #                    help='Parameter initializer to use')
    # model.add_argument('--emb_init', default='normal', type=str,
    #                    help='Parameter initializer to use')
    # model.add_argument('--init_range', type=float, default=0.1,
    #                    help='Parameters initialized by U(-init_range, init_range)')
    # model.add_argument('--emb_init_range', type=float, default=0.01,
    #                    help='Parameters initialized by U(-init_range, init_range)')
    # model.add_argument('--init_std', type=float, default=0.02,
    #                    help='Parameters initialized by N(0, init_std)')
    # model.add_argument('--proj_init_std', type=float, default=0.01,
    #                    help='Parameters initialized by N(0, init_std)')
    # model.add_argument('--primer_square', action='store_true',
    #                    help='Use Primer EZ arch modifications (squared relu)')
    # model.add_argument('--primer_conv', action='store_true',
    #                    help='Use Primer EZ arch modifications (DConv)')
    # model.add_argument('--use_cache', action='store_true',
    #                    help='Whether to return last key/value attentions to speed decoding')
    # model.add_argument('--qat', action='store_true',
    #                    help='Whether to perform Quantization Aware Training (usually based on pretrained model)')

    # opt = parser.add_argument_group('optimizer setup')
    # opt.add_argument('--optim', default='jitlamb', type=str,
    #                  choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],
    #                  help='Optimizer to use')
    # opt.add_argument('--lr', type=float, default=0.01,
    #                  help='Initial learning rate')
    # opt.add_argument('--mom', type=float, default=0.0,
    #                  help='Momentum for sgd')
    # opt.add_argument('--scheduler', default='cosine', type=str,
    #                  choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'cyclic_cosine'],
    #                  help='LR scheduler to use')
    # opt.add_argument('--scheduler_qat', default='cosine', type=str,
    #                  choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'cyclic_cosine'],
    #                  help='LR scheduler to use during QAT')
    # opt.add_argument('--max_step_scheduler', type=int, default=None,
    #                  help='Max number of training steps for LR scheduler')
    # opt.add_argument('--warmup_step', type=int, default=1000,
    #                  help='Number of iterations for LR warmup')
    # opt.add_argument('--decay_rate', type=float, default=0.5,
    #                  help='Decay factor when ReduceLROnPlateau is used')
    # opt.add_argument('--lr_min', type=float, default=0.0,
    #                  help='Minimum learning rate during annealing')
    # opt.add_argument('--clip', type=float, default=0.25,
    #                  help='Gradient clipping')
    # opt.add_argument('--weight_decay', type=float, default=0.0,
    #                  help='Weight decay for adam|lamb')
    # opt.add_argument('--clip_nonemb', action='store_true',
    #                  help='Only clip the gradient of non-embedding params')
    # opt.add_argument('--patience', type=int, default=0,
    #                  help='Patience')
    # opt.add_argument('--eta_min', type=float, default=0.001,
    #                  help='Min learning rate for cosine scheduler')
    # opt.add_argument('--mixed_qat', action='store_true',
    #                  help='Only clip the gradient of non-embedding params')

    # training = parser.add_argument_group('training setup')
    # training.add_argument('--max_step', type=int, default=40000,
    #                       help='Max number of training steps')
    # training.add_argument('--batch_size', type=int, default=256,
    #                       help='Global batch size')
    # training.add_argument('--local_batch_size', type=int, default=None,
    #                       help='Local (per-device) batch size, this setting \
    #                       overrides global --batch_size and sets batch_size \
    #                       to local_batch_size * world_size')
    # training.add_argument('--batch_chunk', type=int, default=1,
    #                       help='Split batch into chunks and train with '
    #                       'gradient accumulation. 16GB V100 FP16 requires 1 chunk, FP32 requires 2 chunks')
    # training.add_argument('--roll', action='store_true',
    #                       help='Enable random shifts within each data stream')
    # training.add_argument('--tgt_len', type=int, default=192,
    #                       help='Number of tokens to predict')
    # training.add_argument('--ext_len', type=int, default=0,
    #                       help='Length of the extended context')
    # training.add_argument('--mem_len', type=int, default=192,
    #                       help='Length of the retained previous heads, number of tokens cached from previous iterations during training')
    # training.add_argument('--seed', type=int, default=42,
    #                       help='Random seed')
    # training.add_argument('--multi_gpu', default=None, type=str,
    #                       choices=['ddp', 'dp'],
    #                       help='Use multiple GPU')
    # training.add_argument('--gpu0_bsz', type=int, default=-1,
    #                       help='Batch size on gpu 0 (for "dp" backend)')
    # # training.add_argument('--same_length', action='store_true',
    # #                       help='Use the same attn length for all tokens')
    # training.add_argument('--varlen', action='store_true',
    #                       help='Use variable length')
    # training.add_argument('--swap_mem', action='store_true',
    #                       help='Swap memory tensors to cpu')

    # val = parser.add_argument_group('validation setup')
    # val.add_argument('--eval_tgt_len', type=int, default=192,
    #                  help='Number of tokens to predict for evaluation')
    # val.add_argument('--eval_batch_size', type=int, default=16,
    #                  help='Eval batch size')
    # val.add_argument('--eval_max_steps', type=int, default=-1,
    #                  help='Max eval steps')
    # val.add_argument('--eval_interval', type=int, default=5000,
    #                  help='Evaluation interval')

    # dist = parser.add_argument_group('distributed setup')
    # dist.add_argument('--local_rank',  type=int,
    #                   default=os.getenv('LOCAL_RANK', 0),
    #                   help='Used for multi-process training.')

    # # post = parser.add_argument_group('post-processing setup')
    # # post.add_argument('--dynamic_quantization', action='store_true',
    # #                   help='Dynamic quantization')
    # # post.add_argument('--post_qat', action='store_true',
    # #                   help='Perform QAT after training the model')

    # parser.set_defaults(**config)
    # args, _ = parser.parse_known_args()

    # args.tied = not args.not_tied


    # Check arguments

    batch_size = new_config['train']['loader']['batch_size']
    tgt_len = new_config['train']['loader']['tgt_len']
    mem_len = new_config['train']['loader']['mem_len']
    ext_len = new_config['train']['loader']['ext_len']

    eval_tgt_len = new_config['eval']['loader']['tgt_len']

    batch_chunk = new_config['train']['batch_chunk']

    if ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')

    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if mem_len == 0:
        if eval_tgt_len > ext_len + tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + ext_len; '
                               f'eval_tgt_len: {eval_tgt_len}, '
                               f'tgt_len: {tgt_len}, '
                               f'ext_len: {ext_len}')
    else:
        if eval_tgt_len > mem_len + tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + mem_len; '
                               f'eval_tgt_len: {eval_tgt_len}, '
                               f'tgt_len: {tgt_len}, '
                               f'mem_len: {mem_len}')

    if batch_size % batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    if new_config['common']['debug'] is None:
        new_config['common']['debug'] = utils.is_debugging()

    # args.config = config_args.config

    return args


def save_checkpoint(conf, model, model_config, optimizer, scheduler, scaler,
                    vocab, epoch, batch, last_iter, train_step, best_val_loss,
                    is_best, work_dir, prefix=''):

    if conf['common']['fp16']:
        amp_state = scaler.state_dict()
    else:
        amp_state = None

    # We never save MixedQAT wrapper, instead we save the fp32 regular model
    if isinstance(model, MixedQATModel):
        model = model.model

    state = {
        'args': conf,
        'model_config': model_config,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'vocab': vocab,
        'amp_state': amp_state,
        'epoch': epoch,
        'batch': batch,
        'last_iter': last_iter,
        'train_step': train_step,
        'best_val_loss': best_val_loss,
        }

    last_chkpt_fname =  prefix + 'checkpoint_last.pt'

    with nv_distributed.sync_workers() as rank:
        last_chkpt_path = os.path.join(work_dir, last_chkpt_fname)
        if rank == 0:
            # always save last checkpoint
            logging.info(f'Saving checkpoint to {last_chkpt_path}')
            torch.save(state, last_chkpt_path)

            # save best checkpoint if better than previous best
            if is_best:
                best_chkpt_fname = prefix + 'checkpoint_best.pt'
                best_chkpt_path = os.path.join(work_dir, best_chkpt_fname)
                logging.info(f'Saving checkpoint to {best_chkpt_path}')
                shutil.copy(last_chkpt_path, best_chkpt_path)

            # save every checkpoint if save_all is true
            if conf['common']['save_all']:
                step_chkpt_fname = f'{prefix}checkpoint_{train_step}.pt'
                step_chkpt_path = os.path.join(work_dir, step_chkpt_fname)
                logging.info(f'Saving checkpoint to {step_chkpt_path}')
                shutil.copy(last_chkpt_path, step_chkpt_path)

def update_dropout(m, dropout_rate):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = dropout_rate


def update_dropatt(m, dropout_rate):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = dropout_rate


def evaluate(eval_iter, model, conf, eval_nomem=True):

    tgt_len = conf['train']['loader']['tgt_len']
    mem_len = conf['train']['loader']['mem_len']
    ext_len = conf['train']['loader']['ext_len']

    eval_tgt_len = conf['eval']['loader']['tgt_len']
    eval_max_steps = conf['eval']['max_steps']
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if mem_len == 0:
        model.reset_length(tgt_len=eval_tgt_len,
                           ext_len=ext_len + tgt_len - eval_tgt_len,
                           mem_len=mem_len
                           )
    else:
        model.reset_length(tgt_len=eval_tgt_len,
                           ext_len=ext_len,
                           mem_len=mem_len + tgt_len - eval_tgt_len,
                           )

    # Evaluation
    total_len, total_loss, total_loss_nomem, steps, total_len_nowarmup, batches = 0, 0., 0., 0, 0, -1
    start_time = time.time()
    with torch.no_grad():
        mems = None
        for batches, (input_ids, labels, seq_len, warm) in enumerate(eval_iter):

            steps += 1
            if eval_max_steps > 0 and i >= eval_max_steps:
                break

            # first with mem
            loss, _, mems, _ = model(input_ids, labels, mems)
            loss = loss.float().mean()
            numel = input_ids.numel()

            # now without mem
            loss_nomem = None
            if eval_nomem:
                loss_nomem, _, _, _ = model(input_ids, labels, None)
                loss_nomem = loss_nomem.float().mean()

            total_len_nowarmup += numel
            if warm:
                # assert (mems is None) or mems.size(1) == model.mem_len
                total_loss += numel * loss.item()
                total_len += numel

                if eval_nomem:
                    total_loss_nomem += numel * loss_nomem.item()

    elapsed = time.time() - start_time

    # Switch back to the training mode
    model.reset_length(tgt_len=tgt_len,
                       ext_len=ext_len,
                       mem_len=mem_len
                       )
    model.train()

    return total_loss, total_len, total_loss_nomem, steps, elapsed, total_len_nowarmup, batches+1


class EvalMetrics:
    def __init__(self, eval_word_count, node_loss, node_len, node_loss_nomem,
                 node_steps, node_elapsed, node_len_nowarmup, batches) -> None:
        node_avg_loss = node_loss / node_len

        self.avg_loss = nv_distributed.all_reduce_item(node_avg_loss, 'mean')
        self.total_loss = nv_distributed.all_reduce_item(node_loss, 'sum')
        self.total_len = nv_distributed.all_reduce_item(node_len, 'sum')
        self.total_len_nowarmup = nv_distributed.all_reduce_item(node_len_nowarmup, 'sum')
        self.total_loss_nomem = nv_distributed.all_reduce_item(node_loss_nomem, 'sum')
        self.total_steps = nv_distributed.all_reduce_item(node_steps, 'sum')
        self.total_elapsed = nv_distributed.all_reduce_item(node_elapsed, 'sum')
        self.eval_word_count = eval_word_count

        self.warmup_discount = 1.0 - float(self.total_len_nowarmup - self.total_len) / self.total_len_nowarmup
        self.word_ppl = math.exp(self.total_loss / (eval_word_count * self.warmup_discount))

        if self.total_loss_nomem is not None:
            avg_loss_nomem = node_loss_nomem / node_len
            self.avg_loss_nomem =  nv_distributed.all_reduce_item(avg_loss_nomem, 'mean')

            self.word_ppl_nomem = math.exp(self.total_loss_nomem /  (eval_word_count * self.warmup_discount))
            self.ppl_nomem = math.exp(self.avg_loss_nomem)
            self.bpc_nomem = self.avg_loss_nomem / math.log(2)
        else:
            self.avg_loss_nomem = None
            self.word_ppl_nomem = None
            self.ppl_nomem = None
            self.bpc_nomem = None

        self.ppl = math.exp(self.avg_loss)
        self.bpc = self.avg_loss / math.log(2)



def train_iteration(model, i, mems, input_ids_chunks, labels_chunks, scaler,
                    optimizer, device, delay_unscale, conf):


    batch_chunk = conf['train']['batch_chunk']
    swap_mem = conf['train']['swap_mem']
    use_fp16 = conf['common']['fp16']
    # trains a given chunk
    cpu = torch.device('cpu')
    input_ids_i = input_ids_chunks[i].contiguous()
    labels_i = labels_chunks[i].contiguous()

    if swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(use_fp16):
        loss, _, mems[i], _ = model(input_ids_i, labels_i, mems[i])
        loss = loss.float().mean().type_as(loss) / batch_chunk

    if swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(cpu, non_blocking=True)

    if use_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    train_loss = loss.float().item()
    return train_loss


def train(train_itr, valid_itr, model, para_model, model_config, optimizer,
          optimizer_sparse, scheduler, scheduler_sparse, scaler, vocab, epoch,
          last_batch, last_iter, train_step, best_val_loss, meters,
          device, conf, valid_file_stats):

    optim_conf = conf['train']['optimizer']
    scheduler_conf = conf['train']['lr_schedule']

    expdir = conf['common']['expdir']
    use_fp16 = conf['common']['fp16']
    log_interval = conf['common']['log_interval']

    dataset_name = conf['dataset']['name']

    use_qat = conf['train']['qat']
    batch_chunk = conf['train']['batch_chunk']
    train_varlen = conf['train']['loader']['varlen']
    train_max_step = conf['train']['max_step']

    eval_interval = conf['eval']['interval']
    no_eval = conf['eval']['no_eval']

    clip_amount = optim_conf['clip']
    inital_lr = optim_conf['lr']

    scheduler_name = scheduler_conf['type_qat'] if use_qat else scheduler_conf['type']
    warmup_step = scheduler_conf['warmup_step']

    # Turn on training mode which enables dropout.
    model.train()

    train_loss = 0
    labels_tokens = 0
    log_step = 0
    log_start_time = time.time()

    mems = [None for _ in range(batch_chunk)]
    # Changes to make train_iter for lm1b to be properly caught
    if dataset_name != 'lm1b':
        if train_varlen:
            train_iter = train_itr.get_varlen_iter(start=last_iter)
        else:
            train_iter = train_itr.get_fixlen_iter(start=last_iter)
    else:
        train_iter = train_itr

    logging.info('Starting training...')
    for batch, (input_ids, labels, seq_len, _) in enumerate(train_iter, start=last_batch+1):
        log_step += 1
        labels_tokens += labels.numel()

        for param in model.parameters():
            param.grad = None

        # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
        input_ids_chunks = torch.chunk(input_ids, batch_chunk, 0)
        labels_chunks = torch.chunk(labels, batch_chunk, 0)

        for i in range(batch_chunk):
            # if this is last chunk and distribued mode then use delay_unscale=True for amp
            if i < batch_chunk - 1 and isinstance(para_model, DistributedDataParallel):
                with para_model.no_sync():
                    train_loss_chunk = train_iteration(
                        para_model, i, mems, input_ids_chunks, labels_chunks, scaler,
                        optimizer, device, True, conf
                    )
            else:
                train_loss_chunk = train_iteration(
                    para_model, i, mems, input_ids_chunks, labels_chunks, scaler,
                    optimizer, device, False, conf
                )

            train_loss += train_loss_chunk

        if use_fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_amount)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_amount)

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            if optimizer_sparse:
                optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if scheduler_name in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < warmup_step:
                curr_lr = inital_lr * train_step / warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if optimizer_sparse:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if scheduler_name == 'cosine':
                    scheduler.step(train_step - warmup_step)
                    if scheduler_sparse:
                        scheduler_sparse.step(train_step - warmup_step)
        elif scheduler_name in ['inv_sqrt', 'cyclic_cosine']:
            scheduler.step(train_step)
            if scheduler_sparse:
                scheduler_sparse.step(train_step)

        if train_step % log_interval == 0:
            cur_loss = train_loss / log_step
            cur_loss = nv_distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0

            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = nv_distributed.all_reduce_item(avg_elapsed, op='max')
            log_start_time = time.time()
            log_step = 0

            lr = optimizer.param_groups[0]['lr']
            throughput = labels_tokens / elapsed
            throughput = nv_distributed.all_reduce_item(throughput, op='sum')
            meters['train_throughput'].update(throughput)
            labels_tokens = 0

            log_str = '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch,
                    train_itr.n_batch,
                    lr,
                    avg_elapsed * 1000,
                    throughput,
                    cur_loss,
                    )

            dllogger_data = {
                'epoch': epoch,
                'train_batch': batch+1,
                'lr': lr,
                'train_time/batch': avg_elapsed * 1000,
                'train_throughput': throughput,
                'train_loss': cur_loss,
                }

            if dataset_name in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                dllogger_data['train_bits_per_character'] = cur_loss / math.log(2)
            else:
                log_str += ' | ppl {:9.2f}'.format(math.exp(cur_loss))
                dllogger_data['train_perplexity'] = math.exp(cur_loss)

            logging.info(log_str)
            dllogger.log(step=tuple([train_step]), data=dllogger_data)

        do_periodic_eval = train_step % eval_interval == 0
        is_final_step = train_step == train_max_step
        interrupted = False #timeout_handler.interrupted

        if (do_periodic_eval or is_final_step or interrupted) and not no_eval:
            eval_start_time = time.time()
            node_metrix = evaluate(valid_itr, model, conf, eval_nomem=False)
            val_metrix = EvalMetrics(valid_file_stats.word_count, *node_metrix)

            logging.info('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| loss {:5.2f} | word ppl {:5.2f}'.format(
                          train_step // eval_interval,
                          train_step,
                          (time.time() - eval_start_time),
                          val_metrix.avg_loss, val_metrix.word_ppl
                          )

            dllogger_data = {
                'valid_elapsed': (time.time() - eval_start_time),
                'valid_loss': val_metrix.avg_loss,
                'valid_ppl': val_metrix.ppl,
                'valid_word_ppl': val_metrix.word_ppl
                }

            if dataset_name in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_metrix.bpc)
                dllogger_data['valid_bits_per_character'] = val_metrix.bpc
            else:
                log_str += ' | ppl {:9.3f}'.format(val_metrix.ppl)
                dllogger_data['valid_perplexity'] = val_metrix.ppl
            logging.info(log_str)
            logging.info('-' * 100)
            dllogger.log(step=tuple([train_step]), data=dllogger_data)

            last_iter = train_itr.last_iter

            # Check if the validation loss is the best we've seen so far.
            is_best = False
            if not best_val_loss or val_metrix.avg_loss < best_val_loss:
                best_val_loss = val_metrix.avg_loss
                is_best = True

            model_to_save = model
            prefix = ''

            if use_qat:
                # Convert the model to a regular FP32 model for saving
                model_float = copy.deepcopy(model)
                model_float = qat_to_float_modules(model_float)
                model_to_save = model_float
                prefix = 'qat_'

            save_checkpoint(conf, model_to_save, model_config, optimizer, scheduler,
                            scaler, vocab, epoch, batch, last_iter,
                            train_step, best_val_loss, is_best,
                            expdir, prefix=prefix)

            # dev-performance based learning rate annealing
            if scheduler_name == 'dev_perf':
                scheduler.step(val_metrix.avg_loss)
                if scheduler_sparse:
                    scheduler_sparse.step(val_metrix.avg_loss)

            # subtract eval time from timers for training
            log_start_time += time.time() - eval_start_time

        if interrupted:
            logging.info(f'Received SIGTERM, exiting')
            sys.exit(0)

        if is_final_step:
            break
    return train_step, best_val_loss


def init(config, disable_multiple_dlogger=False):
    exp_utils.script_init()

    # args = parse_args(config)
    args = None

    dataroot = config['dataset']['dataroot']
    cache_dir = config['dataset']['cache_dir']
    data_dir = config['dataset']['data_dir']

    logdir = config['common']['logdir']
    work_dir = config['common']['expdir']
    use_cuda = config['common']['cuda']
    log_all_ranks = config['common']['log_all_ranks']
    txtlog_file = config['common']['txtlog_file']
    dllog_file = config['common']['dllog_file']
    is_toy = config['common']['toy']
    is_debug = config['common']['debug']
    seed = config['common']['seed']
    no_env = config['common']['no_env']

    pretrained_path = config['train']['pretrained_path']
    local_batch_size = config['train']['loader']['local_batch_size']


    # TODO: below is commented out because nvlm installation issues on Windows
    # if args.affinity != 'disabled':
    #     nproc_per_node = torch.cuda.device_count()
    #     affinity = gpu_affinity.set_affinity(
    #         args.local_rank,
    #         nproc_per_node,
    #         args.affinity
    #     )
    #     print(f'{args.local_rank}: thread affinity: {affinity}')

    # Initialize device and distributed backend
    # torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if use_cuda else 'cpu')
    nv_distributed.init_distributed(use_cuda)

    data_dir = utils.full_path(os.path.join(dataroot, data_dir))

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(data_dir, cache_dir)
    
    cache_dir = utils.full_path(cache_dir, create=True)

    if not os.path.isabs(pretrained_path) and pretrained_path:
        pretrained_path = os.path.join(os.path.dirname(logdir), pretrained_path)
        config['model']['pretrained_path'] = pretrained_path

    
    config['dataset']['cache_dir'] = cache_dir
    config['dataset']['data_dir'] = data_dir
    # args.data, args.work_dir, args.pretrained_path, args.cache_dir, args.dataroot = \
    #     exp_utils.get_create_dirs(args.data, args.dataset, args.experiment_name,
    #                               args.work_dir, args.pretrained_path, args.cache_dir)

    # with nv_distributed.sync_workers() as rank:
    #     if rank == 0:
    #         create_exp_dir(args.work_dir,
    #                        scripts_to_save=[], #['train.py', 'mem_transformer.py'],
    #                        debug=args.debug)

    # Setup logging
    if log_all_ranks:
        log_file = f'train_log_rank_{nv_distributed.get_rank()}.log'
    else:
        log_file = txtlog_file

    log_file = os.path.join(work_dir, log_file)
    dllog_file = os.path.join(work_dir, dllog_file)

    # if args.debug:
    #     log_file = os.devnull
    #     dllog_file = os.devnull

    exp_utils.setup_logging(log_all_ranks=log_all_ranks, filename=log_file)
    exp_utils.setup_dllogger(enabled=True, filename=dllog_file, disable_multiple=disable_multiple_dlogger)

    if is_toy:
        logging.warning('Running in toy mode which means wt2 dataset, only one step training, a lot of batch chunking for laptop GPU')

    if local_batch_size is not None: # default is None
        world_size = nv_distributed.get_world_size()
        config['train']['loader']['batch_size'] = world_size * local_batch_size
        logging.info(f"--local_batch_size was set, adjusting global batch size"
                     f" to {config['train']['loader']['batch_size']} (local_batch_size * world_size)")

    # logging.info(args)
    # dllogger.log(step='PARAMETER', data=vars(args))

    logging.info(f'world size: {nv_distributed.get_world_size()}')

    if not is_debug and not no_env:
        log_env_info()

    #register_ignoring_timeout_handler()

    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # logging.info('=' * 100)
    # for k, v in args.__dict__.items():
    #     logging.info('    - {} : {}'.format(k, v))
    # logging.info('=' * 100)

    return args, device

def load_data(config, device, get_file_stats=True):

    common_conf = config['common']
    dataset_conf = config['dataset']
    train_loader_conf = config['train']['loader']
    eval_loader_conf = config['eval']['loader']

    train_batch_size = train_loader_conf['batch_size']
    tgt_len = train_loader_conf['tgt_len']
    mem_len = train_loader_conf['mem_len']
    ext_len = train_loader_conf['ext_len']

    eval_batch_size = eval_loader_conf['batch_size']
    eval_tgt_len = eval_loader_conf['tgt_len']

    logging.info('Generating/loading dataset...')
    corpus = get_lm_corpus(dataset_conf['data_dir'], dataset_conf['cache_dir'], dataset_conf['name'], dataset_conf['vocab'],
                           vocab_size=dataset_conf['vocab_size'], refresh_cache=common_conf['refresh_cache'])

    if common_conf['no_train']:
        logging.info('Exiting as no training was requested.')
        sys.exit(0)

    if mem_len == 0: # default is 192
        eval_mem_len = 0
    else:
        eval_mem_len = mem_len + tgt_len - eval_tgt_len

    train_itr = corpus.get_iterator('train', train_batch_size, tgt_len,
                                  device=device, ext_len=ext_len)
    valid_itr = corpus.get_iterator('valid', eval_batch_size,
                                  eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=ext_len)
    test_itr = corpus.get_iterator('test', eval_batch_size,
                                  eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=ext_len)

    file_stats = None
    if get_file_stats:
        file_stats = corpus.file_stats()
        for file_stat in file_stats:
            logging.info(file_stat)

    return  corpus.vocab, train_itr, valid_itr, test_itr, file_stats


def create_or_load_model(config, args, device, ntokens)->Tuple[ArchaiModel, dict]:

    train_config = config['train']
    model_config = config['model']
    dataset_conf = config['dataset']

    mixed_qat = train_config['mixed_qat']
    qat = train_config['qat']
    pretrained_path = train_config['pretrained_path']
    dataset_name = dataset_conf['name']

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [] # head cluster projection is never tied with embeddings
    if model_config['adaptive']:
        assert dataset_name in ['wt103', 'wt2', 'lm1b'] or dataset_conf['name'].startswith('olx_')
        if dataset_name in ['wt103', 'wt2'] or dataset_name.startswith('olx_'):
            cutoffs = [19997, 39997, 199997, ntokens]
            tie_projs = [False] + [True] * (len(cutoffs)-1)
        elif dataset_name == 'lm1b':
            cutoffs = [59997, 99997, 639997, ntokens]
            tie_projs = [False] + [False] * (len(cutoffs)-1)
        else:
            raise RuntimeError(f'Dataset {dataset_name} not supported for set cutoffs and tie_projs')


    m_config = copy.deepcopy(model_config)

    # Adds additional info for logging model info
    m_config['cutoffs'] = cutoffs
    m_config['n_token'] = ntokens
    m_config['dtype'] = None
    m_config['tie_projs'] = tie_projs
    m_config['sample_softmax'] = train_config['optimizer']['sample_softmax']
    m_config['tgt_len'] = train_config['loader']['tgt_len']
    m_config['ext_len'] = train_config['loader']['ext_len']
    m_config['mem_len'] = train_config['loader']['mem_len']

    print(f'Model config: {m_config}')

    # model_config = {
    #     'n_token': ntokens,
    #     'n_layer': args.n_layer,
    #     'n_head': args.n_head,
    #     'd_model': args.d_model,
    #     'd_head': args.d_head,
    #     'd_inner': args.d_inner,
    #     'dropout': args.dropout,
    #     'dropatt': args.dropatt,
    #     'dtype': None,
    #     'tie_weight': args.tied,
    #     'd_embed': args.d_embed,
    #     'div_val': args.div_val,
    #     'tie_projs': tie_projs,
    #     'pre_lnorm': args.pre_lnorm,
    #     'tgt_len': args.tgt_len,
    #     'ext_len': args.ext_len,
    #     'mem_len': args.mem_len,
    #     'cutoffs': cutoffs,
    #     'adaptive': args.adaptive,
    #     'same_length': args.same_length,
    #     'attn_type': args.attn_type,
    #     'clamp_len': args.clamp_len,
    #     'sample_softmax': args.sample_softmax,

    #     'weight_init_type': args.init,
    #     'weight_init_range': args.init_range,
    #     'weight_init_std': args.init_std,
    #     'proj_init_std': args.proj_init_std,

    #     'primer_square': args.primer_square,
    #     'primer_conv': args.primer_conv,
    #     'use_cache': args.use_cache
    #     }

    if qat and not pretrained_path:
        logging.warning('QAT usually starts from a pretrained model. Check the --pretrained_path argument.')

    if qat and mixed_qat:
        raise ValueError('QAT and Mixed QAT cannot be used at the same time.')

    if pretrained_path:
        logging.info('Overwriting the provided model config with the pretrained model config.')
        model, m_config, _ = load_model_from_checkpoint(model_config['model_type'], pretrained_path, on_cpu=False)
    else:
        model = load_model_from_config(m_config['model_type'], m_config)

    if mixed_qat:
        model = MixedQATModel(model)

    n_params = model.get_params()
    n_all_param = n_params['total']
    n_nonemb_param = n_params['non_embedding']
    logging.info('#params = {}'.format(n_all_param))
    logging.info('#non emb params = {}'.format(n_nonemb_param))

    if qat:
        model = prepare_with_qat(model, onnx_compatible=True)

    return model, m_config

def create_optimizer(optim_conf, model):
    # optimizer
    optim_type = optim_conf['type']
    sample_softmax = optim_conf['sample_softmax']
    lr = optim_conf['lr']
    weight_decay = optim_conf['weight_decay']
    momentum = optim_conf['momentum']

    if optim_type.lower() == 'sgd':
        if sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=lr * 2)
            optimizer = optim.SGD(dense_params, lr=lr, momentum=momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=momentum)
            optimizer_sparse = None
    elif optim_type.lower() == 'adam':
        if sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=lr)
            optimizer = optim.Adam(dense_params, lr=lr,
                                   weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
            optimizer_sparse = None
    elif optim_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
        optimizer_sparse = None
    elif optim_type.lower() == 'lamb':
        optimizer = lamb_optimizer.Lamb(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        optimizer_sparse = None
    elif optim_type.lower() == 'jitlamb':
        optimizer = lamb_optimizer.JITLamb(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
        optimizer_sparse = None
    else:
        raise NotImplementedError(f'Optimizer {optim_type} is not implemented')

    return optimizer, optimizer_sparse


def create_grad_scaler(use_fp16, model, optimizer):
    scaler = None
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    return scaler


def distributed_model(conf, args, model, device):
    # by default this argument is not used, instead we spawn multiple instances
    # using command line:
    # python -m torch.distributed.launch --nproc_per_node="$2" train.py \
    #         --config_file wt103_base.yaml \
    #         "${@:3}"

    multi_gpu = conf['common']['multi_gpu']
    batch_chunk = conf['train']['batch_chunk']
    gpu0_bsz = conf['train']['gpu0_bsz']

    if multi_gpu == 'ddp' and torch.distributed.is_initialized():
        para_model = DistributedDataParallel(model,
                                             device_ids=[args.local_rank],
                                             output_device=args.local_rank,
                                             broadcast_buffers=False,
                                             find_unused_parameters=utils.is_debugging(),
                                             )
    elif multi_gpu == 'dp':
        if gpu0_bsz >= 0:
            para_model = BalancedDataParallel(gpu0_bsz // batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model

    return para_model, model


def create_scheduler(train_conf, optimizer, optimizer_sparse):

    scheduler_conf = train_conf['lr_schedule']

    scheduler, scheduler_sparse = None, None
    scheduler_name = scheduler_conf['type_qat'] if train_conf['qat'] else scheduler_conf['type']
    max_step_scheduler = scheduler_conf['max_step_scheduler']
    warmup_step = scheduler_conf['warmup_step']
    patience = scheduler_conf['patience']
    eta_min = scheduler_conf['eta_min']
    lr_min = scheduler_conf['lr_min']
    decay_rate = scheduler_conf['decay_rate']

    sample_softmax = train_conf['optimizer']['sample_softmax']
    lr = train_conf['optimizer']['lr']

    # scheduler
    if scheduler_name == 'cosine':
        if max_step_scheduler:
            max_step = max_step_scheduler
        else:
            max_step = train_conf['max_step']

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_step - warmup_step, eta_min=eta_min)
        if sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_sparse, max_step - warmup_step,
                eta_min=eta_min)
        else:
            scheduler_sparse = None
    elif scheduler_name == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > warmup_step \
                    else step / (warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.LambdaLR(
                optimizer_sparse,
                lr_lambda=lr_lambda
                )
        else:
            scheduler_sparse = None
    elif scheduler_name == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=decay_rate, patience=patience,
            min_lr=lr_min,
            )
        if sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_sparse, factor=decay_rate, patience=patience,
                min_lr=lr_min,
                )
        else:
            scheduler_sparse = None
    elif scheduler_name == 'cyclic_cosine':
        init_decay_epochs = int((max_step-warmup_step) / 2)
        restart_interval = int((max_step-warmup_step) / 4)

        scheduler = CyclicCosineDecayLR(optimizer, init_decay_epochs, eta_min, restart_interval, 
                                        warmup_epochs=warmup_step, warmup_start_lr=lr*0.01)
        if sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = CyclicCosineDecayLR(optimizer_sparse, init_decay_epochs, eta_min, restart_interval, 
                                        warmup_epochs=warmup_step, warmup_start_lr=lr*0.01)
        else:
            scheduler_sparse = None
    elif scheduler_name == 'constant':
        pass

    return scheduler, scheduler_sparse


def train_main(conf, device, train_itr, valid_itr, model, para_model, model_config,
                optimizer, optimizer_sparse, scheduler,
                scheduler_sparse, scaler, vocab, valid_file_stats):

    roll_seed = conf['common']['seed']
    expdir = conf['common']['expdir']
    use_fp16 = conf['common']['fp16']
    restart = conf['common']['restart']

    train_max_step = conf['train']['max_step']
    mem_len = conf['train']['loader']['mem_len']
    tgt_len = conf['train']['loader']['tgt_len']
    use_roll = conf['train']['loader']['roll']

    model_type = conf['model']['model_type']
    dropout_rate = conf['model']['dropout']
    dropatt_rate = conf['model']['dropatt']

    dynamic_quant = conf['post_processing']['dynamic_quantization']

    train_step = 0
    start_epoch = 1
    last_batch = 0
    last_iter = 0
    best_val_loss = None

    if restart:
        try:
            model, model_config, checkpoint = load_model_from_checkpoint(model_type, restart, on_cpu=False)
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            if use_fp16:
                scaler.load_state_dict(checkpoint['amp_state'])
            train_step = checkpoint['train_step']
            start_epoch = checkpoint['epoch']
            last_batch = checkpoint['batch']
            last_iter = checkpoint['last_iter']
            best_val_loss = checkpoint['best_val_loss']

            if train_step >= train_max_step:
                logging.info(f'Loaded checkpoint after {train_step} steps, but '
                             f'this run was scheduled for a total of '
                             f'{train_max_step} steps, exiting')
                sys.exit(1)

            model.apply(functools.partial(update_dropout, dropout_rate))
            model.apply(functools.partial(update_dropatt, dropatt_rate))
        except FileNotFoundError:
            logging.info(f'Could not load checkpoint from {restart}, '
                         f'starting training from random init')

    meters = {}
    warmup = mem_len // tgt_len + 2
    meters['train_throughput'] = AverageMeter(warmup=warmup)
    ###########################################################################
    # Train
    ###########################################################################
    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    start_time = time.time()
    try:
        for epoch in itertools.count(start=start_epoch):
            if use_roll: # enable random shifts in datasets
                train_itr.roll(seed=roll_seed + epoch)
            train_step, best_val_loss = train(
                train_itr, valid_itr, model, para_model, model_config,
                optimizer, optimizer_sparse, scheduler,
                scheduler_sparse, scaler, vocab, epoch, last_batch,
                last_iter, train_step, best_val_loss, meters,
                device, conf, valid_file_stats
                )

            last_batch = 0
            last_iter = 0

            if train_step == train_max_step:
                logging.info('-' * 100)
                logging.info('End of training')
                break

        if dynamic_quant:
            dynamic_quantization_torch_from_model(model.cpu())

            save_checkpoint(conf, model, model_config, optimizer, scheduler,
                            scaler, vocab, epoch, last_batch, last_iter,
                            train_step, best_val_loss, False,
                            expdir, prefix='qnt-')

    except KeyboardInterrupt:
        logging.info('-' * 100)
        logging.info('Exiting from training early')

    elapsed = time.time() - start_time

    return elapsed, best_val_loss, meters


def evaluate_main(conf, model, checkpoint_path:str, test_itr, test_file_stats):

    model_type = conf['model']['model_type']
    no_eval = conf['eval']['no_eval']

    dataset_name = conf['dataset']['name']

    n_params = model.get_params()
    summary = {
        'n_all_param': n_params['total'],
        'n_nonemb_param': n_params['non_embedding']
    }

    if not no_eval and os.path.exists(checkpoint_path):
        # Load the best saved model
        model, _, _ = load_model_from_checkpoint(model_type, checkpoint_path, on_cpu=False)

        # Run on test data
        test_start_time = time.time()
        node_metrix = evaluate(test_itr, model, conf, eval_nomem=True)
        test_metrix = EvalMetrics(test_file_stats.word_count, *node_metrix)

        test_elapsed = time.time() - test_start_time

        logging.info('=' * 100)
        if dataset_name in ['enwik8', 'text8']:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | word ppl {:9.3f} | test bpc {:9.5f}'.format(
                test_elapsed, test_metrix.avg_loss, test_metrix.word_ppl, test_metrix.bpc))
        else:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | word ppl {:9.3f} | test ppl {:9.3f}'.format(
                test_elapsed, test_metrix.avg_loss, test_metrix.word_ppl, test_metrix.ppl))
        logging.info('=' * 100)

        summary.update({
            'test_word_count': test_metrix.eval_word_count,
            'test_total_elapsed': test_metrix.total_elapsed,
            'test_elapsed': test_elapsed,
            'test_total_loss': test_metrix.total_loss,
            'test_total_loss_nomem': test_metrix.total_loss_nomem,
            'test_avg_loss': test_metrix.avg_loss,
            'test_avg_loss_nomem': test_metrix.avg_loss_nomem,
            'test_steps': test_metrix.total_steps,
            'test_len': test_metrix.total_len,
            'total_len_nowarmup': test_metrix.total_len_nowarmup,
            'warmup_discount': test_metrix.warmup_discount,
            'test_word_ppl': test_metrix.word_ppl,
            'test_word_ppl_nomem': test_metrix.word_ppl_nomem
            })

        if dataset_name in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_metrix.bpc
            summary['test_bits_per_character_nomem'] = test_metrix.bpc_nomem
        else:
            summary['test_ppl'] = test_metrix.ppl
            summary['test_ppl_nomem'] = test_metrix.ppl_nomem

    return summary


def main():

    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    args, unknown = parser.parse_known_args()

    # Loads config file
    conf = create_conf(param_args=sys.argv, use_args=True)
    Config.set_inst(conf)

    # create experiment dir
    create_dirs(conf, False)

    print(conf)


    # get command line args
    args, device = init(conf)

    use_qat = conf['train']['qat']
    exp_dir = conf['common']['expdir']
    experiment_name = conf['common']['experiment_name']

    model_type = conf['model']['model_type']

    dataset_name = conf['dataset']['name']
    vocab_type = conf['dataset']['vocab']

    post_qat = conf['post_processing']['qat']


    # load tokenizer and datasets
    vocab, train_itr, valid_itr, test_itr, file_stats = load_data(conf, device)

    # create model
    ntokens = len(vocab)
    model, model_config = create_or_load_model(conf, args, device, ntokens)

    # create optimizer
    optimizer, optimizer_sparse = create_optimizer(conf['train']['optimizer'], model)

    model = model.to(device)

    # create gradient scaler
    scaler = create_grad_scaler(conf['common']['fp16'], model, optimizer)

    # enable distributed training
    para_model, model = distributed_model(conf, args, model, device)

    # create scheduler
    scheduler, scheduler_sparse = create_scheduler(conf['train'], optimizer, optimizer_sparse)

    training_time, best_val_loss, meters = train_main(conf, device, train_itr, valid_itr, model, para_model,
        model_config, optimizer, optimizer_sparse, scheduler,
        scheduler_sparse, scaler, vocab, file_stats[1])

    checkpoint_path = os.path.join(exp_dir, 'checkpoint_best.pt' if not use_qat else 'qat_checkpoint_best.pt')
    summary = evaluate_main(conf, model, checkpoint_path, test_itr, file_stats[-1])

    logging.info(f'Training time: {(training_time / 60):.2f} minutes')
    logging.info(f'Training throughput: {meters["train_throughput"].avg:.2f} tok/s')

    input_ids, *_ = next(iter(valid_itr))
    model.to('cpu')
    input_ids = input_ids[:1,:].to('cpu') # make it batch size of one
    pt_ops_mem, pt_ops_time, pt_ops_flops, pt_inf_time = ml_perf_utils.inference_stats(model, input_ids=input_ids, labels=None, mems=None)
    _, process_mem = ml_perf_utils.model_memory(
        lambda: load_model_from_checkpoint(model_type, checkpoint_path, on_cpu=True))

    summary.update({
        'experiment_name': experiment_name,
        'run_date': str(datetime.now()),
        'input_ids.shape(0)': input_ids.shape[0],
        'input_ids.shape(1)': input_ids.shape[1],
        'dataset': dataset_name,
        'vocab_size': ntokens,
        'vocab_type': vocab_type,
        'train_throughput': meters['train_throughput'].avg,
        'train_elapsed': training_time / 60.0,
        'valid_loss': best_val_loss,
        'valid_ppl': math.exp(best_val_loss) if best_val_loss else None,
        'n_token': ntokens,
        'n_layer': model_config['n_layer'],
        'n_head': model_config['n_head'],
        'd_model': model_config['d_model'],
        'd_head': model_config['d_head'],
        'd_inner': model_config['d_inner'],
        'dropatt': model_config['dropatt'],
        'd_embed': model_config['d_embed'],
        'div_val': model_config['div_val'],
        'tgt_len': model_config['tgt_len'],
        'ext_len': model_config['ext_len'],
        'mem_len': model_config['mem_len'],
        'cutoffs': model_config['cutoffs'],
        'primer_conv': model_config['primer_conv'],
        'primer_square': model_config['primer_square'],
        'pt_ops_mem': pt_ops_mem,
        'pt_ops_time_us': pt_ops_time,
        'pt_ops_flops': pt_ops_flops,
        'pt_inf_time_us': pt_inf_time,
        'process_mem': process_mem
        })
    summary.update((k, '') for k, v in summary.items() if v is None)

    logging.info(pprint.pformat(summary))

    utils.save_as_yaml(summary, os.path.join(exp_dir, 'summary.yaml'))
    utils.save_as_yaml(model_config, os.path.join(exp_dir, 'model_config.yaml'))

    summary_csv_filepath = os.path.join(exp_dir, 'summaries.tsv')
    with nv_distributed.sync_workers() as rank:
        if rank == 0:
            utils.append_csv_file(summary_csv_filepath, list(summary.items()))

    logging.info(f'Output dir: {exp_dir}')
    dllogger.log(step=tuple(), data=summary)

    if post_qat:
        # Creates a dictionary of replacement configs
        replace_model_config = {
            'dropout': 0.0,
            'dropatt': 0.0
        }

        # Loads the model from the best pre-trained checkpoint
        model, model_config, _ = load_model_from_checkpoint(model_type, checkpoint_path, replace_model_config=replace_model_config, on_cpu=False)

        # Prepares the model with QAT (also allows for distributed training)
        model = prepare_with_qat(model, onnx_compatible=True)
        model = model.to(device)
        para_model, model = distributed_model(args, model, device)

        # QAT-based arguments
        conf['common']['restart'] = None
        conf['train']['qat'] = True
        conf['train']['max_step'] = 10000
        conf['train']['optimizer']['lr'] = conf['train']['optimizer']['lr'] / 100
        conf['train']['lr_schedule']['eta_min'] = conf['train']['lr_schedule']['eta_min'] / 100
        conf['eval']['interval'] = 1000
        conf['train']['lr_schedule']['warmup_step'] = 1000
        conf['train']['optimizer']['type'] = 'adam'

        # re-create optimizer
        optimizer, optimizer_sparse = create_optimizer(conf['train']['optimizer'], model)

        # re-create scheduler
        scheduler, scheduler_sparse = create_scheduler(conf['train'], optimizer, optimizer_sparse)

        # Performs a QAT fine-tuning
        training_time, best_val_loss, meters = train_main(conf, device, train_itr, valid_itr, model, para_model,
                                                          model_config, optimizer, optimizer_sparse, scheduler,
                                                          scheduler_sparse, scaler, vocab, file_stats[1])


if __name__ == "__main__":
    main()
