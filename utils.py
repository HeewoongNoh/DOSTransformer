from cmath import nan
import numpy as np
import argparse

import torch

# return (mean_correlation, minimum correlation, maximum correlation)
def correlation(y, preds):

    numerator = torch.matmul(preds - preds.mean(dim = 1).reshape(-1, 1), (y - y.mean(dim = 1).reshape(-1, 1)).T).diag()
    # nan_numerator = torch.isnan(numerator)
    # for i in nan_numerator:
    #     if i == True:
    #         print("numerator_nan")
    # inf_numerator = torch.isinf(numerator)
    # for i in inf_numerator:
    #     if i == True:
    #         print("numerator_inf")

    denominator1 = torch.norm(preds - preds.mean(dim = 1).reshape(-1, 1), dim = 1)
    # nan_denominator1 = torch.isnan(denominator1)
    # for i in nan_denominator1:
    #     if i == True:
    #         print("denom1_nan")
            
    # inf_denominator1 = torch.isinf(denominator1)
    # for i in inf_denominator1:
    #     if i == True:
    #         print('denom1_inf')

    denominator2 = torch.norm(y - y.mean(dim = 1).reshape(-1, 1), dim = 1)
    # nan_denominator2 = torch.isnan(denominator2)
    # for i in nan_denominator2:
    #     if i == True:
    #         print("denom2_nan")
    # inf_denominator2 = torch.isinf(denominator2)
    # for i in inf_denominator2:
    #     if i == True:
    #         print("denom_inf")

    output = 1 - numerator / (denominator1 * denominator2)
    # nan_output = torch.isnan(output)
    # for i in nan_output:
    #     if i == True:
    #         print("output_nan")
    # inf_output = torch.isinf(output)
    # for i in inf_output:
    #     if i == True:
    #         print("output_inf")
            
    output = output.mean()
    return output


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", '-d', type=int, default=6, help="GPU to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size") # (16 --> 2000) (64 --> 6000)
    parser.add_argument("--layers", "-l", type=int, default=3, help="The number layers of the Processor")
    parser.add_argument("--transformer", "-t", type=int, default=2, help="The number of Transformer layers")
    parser.add_argument("--eval", type=int, default=5, help="evaluation step")
    parser.add_argument("--es", type=int, default=50, help="Early Stopping Criteria")
    parser.add_argument("--embedder", type=str, default="gntransformer", help="Early Stopping Criteria")
    parser.add_argument("--hidden", type=int, default=256, help="Early Stopping Criteria")
    parser.add_argument("--masking_ratio", type=float, default=0.1, help="Node masking ratio in pretraining task")
    parser.add_argument("--spark", type=float, default=0.1, help="Spark criterion")
    parser.add_argument("--preepoch", type=int, default=60, help="Pre train epoch")
    parser.add_argument("--num_e3nn_layer", type=int, default=3, help="Number of e3nn layers")
    parser.add_argument("--mul", type=int, default=32, help="multiplicity of irreducible representations")
    parser.add_argument("--r_max", type=int, default=4, help="cutoff radius for convolution")
    parser.add_argument("--pretrain", type=str, default='efermi', help="pretraining_task")
    parser.add_argument("--train_joint", action='store_true', help='Train joint')
    parser.add_argument("--alpha", type=float, default=1.0, help = 'alpha for the spark loss1')
    parser.add_argument("--beta", type=float, default=0.25, help = 'alpha for the spark loss2')
    parser.add_argument("--sampling_ratio", type=int, default = 1, help='Sampling ratio of sparked energy' )
    parser.add_argument("--window_size", type = int, default = 2, help = 'Window size of embedding')
    parser.add_argument("--random_state", type=int, default=0, help = 'Random state for dataset split')
    parser.add_argument("--dataset", type=str, default='whole', help = 'Dos dataset2,3,4')
    parser.add_argument("--OOD", action ='store_true', help = 'Inference model for OOD')
    parser.add_argument("--pred", type=str,default='gt', help = "preds(dos) from what models for bandgap prediction")
    parser.add_argument("--decay", type=float,default=0.05, help = "Adamw weight decay")
    return parser.parse_args()

def training_config(args):
    train_config = dict()
    for arg in vars(args):
        train_config[arg] = getattr(args,arg)
    return train_config

def pred_bandgap(train_config):
    name = ''
    dic = train_config
    config = ["random_state","embedder", "lr", "batch_size", "hidden"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name
def get_name(train_config):
    name = ''
    dic = train_config
    config = ["masking_ratio","preepoch","random_state","layers", "dataset", "lr", "batch_size", "hidden"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def train_joint_get_name(train_config):
    name = ''
    dic = train_config
    config = ["layers", "transformer", "lr", "batch_size", "hidden", "spark"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def pretrain_masking_get_name(train_config):
    name = ''
    dic = train_config
    config = ["layers", "lr", "batch_size", "hidden", "masking_ratio"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def pretrain_get_name2(train_config):
    name = ''
    dic = train_config
    config = ["masking_ratio","preepoch", "lr", "batch_size"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name



def pretrain_get_name(train_config):
    name = ''
    dic = train_config
    config = ["lr", "batch_size", "hidden","embedder"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name


def finetune_get_name(train_config):
    name = ''
    dic = train_config
    config = ["layers", "transformer", "lr", "batch_size", "hidden", "preepoch"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name


def finetune_joint_get_name(train_config):
    name = ''
    dic = train_config
    config = ["layers", "transformer", "lr", "batch_size", "hidden", "spark", "preepoch"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name


def euclidean_get_name(train_config):
    name = ''
    dic = train_config
    config = ["num_e3nn_layer","hidden","random_state","lr", "batch_size","r_max","mul"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def mlp_get_name(train_config):
    name = ''
    dic = train_config
    config = ["embedder", "random_state","dataset","layers", "lr", "batch_size", "hidden"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name


def exp_get_name1(train_config):
    name = ''
    dic = train_config
    config = ["transformer", "layers", "lr", "batch_size", "hidden","pretrain","spark","preepoch","sampling_ratio","window_size"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def exp_get_name2(train_config):
    name = ''
    dic = train_config
    config = ["transformer", "layers", "lr", "batch_size", "hidden","pretrain","preepoch","sampling_ratio"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def exp_get_name3(train_config):
    name = ''
    dic = train_config
    config = ["transformer", "layers", "lr", "batch_size", "hidden","pretrain","spark","preepoch","alpha","sampling_ratio"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def exp_get_name5(train_config):
    name = ''
    dic = train_config
    config = ["embedder", "lr", "batch_size", "hidden","pretrain","preepoch","random_state"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def exp_get_name6(train_config):
    name = ''
    dic = train_config
    config = ["embedder", "lr", "batch_size", "hidden","random_state"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name

def exp_get_name7(train_config):
    name = ''
    dic = train_config
    config = ["transformer", "layers","embedder", "lr", "batch_size", "hidden","random_state","decay"]

    for key in config:
        a = f'{key}'+f'({dic[key]})'+'_'
        name += a
    return name