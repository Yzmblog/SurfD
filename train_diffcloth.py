# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from utils.logger import setup_logger
from utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from utils.miscellaneous import mkdir, set_seed
from training_loop_single import TrainLoop
from models.cfg_sampler import ClassifierFreeSampleModel

from utils.model_util import create_model_and_diffusion

from data_loaders.dataset import UDFs3d
import os
import torch
import numpy as np

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    return batch_sampler

def main():
    global logger
    args = train_args()
    fixseed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    if args.distributed:
        print("Init distributed training on local rank {} ({}), world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        args.local_rank = int(os.environ["LOCAL_RANK"])
        synchronize()
    logger = setup_logger("Diffcloth", args.save_dir, get_rank())

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        args4print = args
        args4print.device = str(args4print.device)
        json.dump(vars(args4print), fw, indent=4, sort_keys=True)
    logger.info("Using {} GPUs".format(args.num_gpus))

    logger.info("creating data loader...")


    name = args.dataset
    dset_root = args.data_dir
    dset_category = 'train'
    dataset_train = UDFs3d(name, dset_root, dset_category, args.cond_mode)

    shuffle = True
    
    args.batch_size = 2 #4
    images_per_gpu = args.batch_size
    images_per_batch = images_per_gpu * get_world_size()
    iters_per_batch = len(dataset_train) // images_per_batch
    num_iters = args.num_steps
    start_iter = 0
    logger.info("Train with {} images per GPU.".format(images_per_gpu))
    logger.info("Total batch size {}".format(images_per_batch))
    logger.info("Total training steps {}".format(num_iters))
 
    sampler = make_data_sampler(dataset_train, shuffle, args.distributed)
    
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, args.batch_size, shuffle=True, num_workers=6,
        pin_memory=True,
    )
    if args.unconstrained:
        args.num_actions = None
    

    logger.info("creating model and diffusion...")
    diff_model, diffusion = create_model_and_diffusion(args)
   
    if args.guidance_param != 1:
        print('classifier free')
        diff_model = ClassifierFreeSampleModel(diff_model)   # wrapping model with the classifier-free sampler

    print(args.guidance_param)
    if args.distributed:
        args.gpu = args.local_rank
        args.world_size = torch.distributed.get_world_size()
        diff_model = diff_model.to(args.local_rank)
     
        diff_model = torch.nn.parallel.DistributedDataParallel(
            diff_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    else:
        diff_model = diff_model.cuda()

    inds = np.random.choice(100000, 10000, replace=False)
    TrainLoop(args, diff_model, diffusion, dataloader_train, logger).run_loop(inds=inds)




if __name__ == "__main__":
    main()
