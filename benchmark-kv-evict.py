#!/usr/bin/env python3
from functools import partial
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
import json
from timm.models import create_model, is_model, list_models
from timm.utils import AverageMeter, setup_default_logging

from patch import patch_model_for_kv_eviction

# import natten

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--num-warmup', default=25, type=int)
parser.add_argument('--num-iters', default=200, type=int)
parser.add_argument('--log-freq', default=500, type=int)
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument("--evict-algo", default="topk", type=str)
parser.add_argument("--evict-k", default=0, type=int)
parser.add_argument("--evict-start", default=0, type=float)
parser.add_argument("--evict-end", default=0, type=float)
parser.add_argument("--evict-num", default=0, type=int)
parser.add_argument("--evict-after-end", default=-1, type=int)
parser.add_argument("--evict-step", default=0, type=int)
parser.add_argument("--savedir", default=None, type=str)


def validate(args):
    amp_autocast = suppress  # do nothing
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
    # create model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=3)
    
    if args.evict_k > 0:
        model = patch_model_for_kv_eviction(
            model, 
            algorithm=args.evict_algo,
            k=args.evict_k,
            to_evict=args.evict_num,
            start_layer=args.evict_start,
            end_layer=args.evict_end,
            after_end=args.evict_after_end,
            step=args.evict_step
        )

    model = model.cuda()
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    print(f'Model {args.model} created')

    #if args.channels_last:
    model = model.to(memory_format=torch.channels_last)

    batch_time = AverageMeter()
    num_warmup = args.num_warmup
    num_iters = args.num_iters

    model.eval()
    #model.train() 
    #with torch.no_grad():
    # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
    if args.model.endswith("224"):
        image_size = 224
    elif args.model.endswith("336"):
        image_size = 336
    elif args.model.endswith("384"):
        image_size = 384
    else:
        image_size = args.image_size
    input = torch.randn((args.batch_size, 3, image_size, image_size)).cuda()
    input = input.contiguous(memory_format=torch.channels_last)
    
    for i in range (num_warmup):
        with amp_autocast():
            model(input)
    
    #model.to(torch.float16)
    #input = input.to(torch.float16)        

    with torch.no_grad(): 
        for batch_idx in range(num_iters):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            with amp_autocast():
                output = model(input)
            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)/1000

            # measure elapsed time
            if batch_idx > 0 and batch_idx < num_iters - 1:
                batch_time.update(elapsed)

                if batch_idx % args.log_freq == 0:
                    print(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)'.format(
                            batch_idx, num_iters, batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg))

    timea = batch_time.avg
    results = OrderedDict(
        rate=round(args.batch_size / timea, 4))

    print(' * Throughput {:.3f} '.format(
       results['rate']))
    
    if args.savedir is not None:
        os.makedirs(args.savedir, exist_ok=True)
        model_str = args.model.replace(".", "_")
        results_file = os.path.join(
            args.savedir,
            f"{model_str}-algo_{args.evict_algo}-k{args.evict_k}-num{args.evict_num}.json"
        )
        write_results(results_file, results, format="json") 


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
