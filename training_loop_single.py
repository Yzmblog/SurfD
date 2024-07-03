import functools
import os
import time
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from utils.comm import is_main_process

from diffusion.resample import create_named_schedule_sampler


from torch.utils.tensorboard import SummaryWriter

from utils.utils import random_point_sampling

from AutoEncoder.models.dgcnn import Dgcnn
import clip

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, model, diffusion, data, logger=None):
        self.args = args
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.grid_size = args.grid_size
        self.logger = logger


        print(args.clip_value)
        print("Apply clip: ", args.clip_value)

        self.model = model
        self.diffusion = diffusion
        if args.distributed:
            self.cond_mode = model.module.cond_mode
            self.clip_version = model.module.clip_version
        else:
            self.clip_version = model.clip_version
            self.cond_mode = model.cond_mode

        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        print('num_actions: ', args.num_actions)
        print('dataloader length: ', len(self.data))
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        print("batch_size:", self.batch_size)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model
        self.log_writer = SummaryWriter(self.save_dir+"/logs") # added.

        if 'text' or 'img' in args.cond_mode:
            latent_size = 64
        else:
            latent_size = 32
        encoder = Dgcnn(latent_size)
        self.encoder = encoder.eval()
        ckpt = torch.load(args.ae_dir)
        self.encoder.load_state_dict(ckpt["encoder"], strict=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        if args.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.vqvae = self.vqvae.to(device=self.local_rank)
            self.device = self.local_rank
        else:
            self.encoder = self.encoder.to(device=self.args.device)

        if 'sketch' in self.cond_mode or 'img' in self.cond_mode:
            print('EMBED SKETCH IMAGE')
            print('Loading CLIP...')
            self.clip_model = self.load_and_freeze_clip(self.clip_version)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            self.logger.info(f"loading model from checkpoint: {resume_checkpoint}...")
            
            sd = dist_util.load_state_dict(
                    resume_checkpoint, map_location="cpu"
                )

            self.model.load_state_dict({k:v for k, v in sd.items()}, 
                                       strict=False
            )

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def worker_init_fn(self, worker_id):
        np.random.seed(int(time.time() * 10000000) % 10000000 + worker_id)


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            self.logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def randbool(self, *size):
        return torch.randint(2, size) == torch.randint(2, size)
    def run_loop(self, inds=None):
        loss_L1 = torch.nn.L1Loss().cuda(self.device)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')

            for i, batch in enumerate(self.data):

                if 'sketch' in self.cond_mode or 'img' in self.cond_mode:
                    _, _, pcds, coords, gt_udf, gt_grad, img = batch
                elif 'text' in self.cond_mode:
                    _, _, pcds, coords, gt_udf, gt_grad, text = batch
                elif 'category' in self.cond_mode:
                    _, _, pcds, coords, gt_udf, gt_grad, label = batch
                else:
                    _, _, pcds, coords, gt_udf, gt_grad = batch
                
                loss_args = {}
                pcds = pcds.cuda()

                num_points_pcd = 10000
                pcds = random_point_sampling(pcds, num_points_pcd, inds=inds)
                latent_codes = self.encoder(pcds).unsqueeze(1)


                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                cond = {}
                cond["y"] = {}
                if self.cond_mode == "no_cond":
                    cond['y']['mask'] = torch.ones(latent_codes.shape, dtype=torch.bool).cuda()
                elif self.cond_mode == "category":
                    cond['y']['action'] = torch.tensor(label, dtype=torch.float).cuda()
                    cond['y']['action_text'] = torch.tensor(label, dtype=torch.int64).cuda()
                elif self.cond_mode == 'sketch' or self.cond_mode == 'img':
                    cond['y']['context'] = self.clip_model.encode_image(img).cuda()
                elif self.cond_mode == 'text':
                    cond['y']['text'] = text
                    cond['y']['scale'] = self.args.guidance_param * torch.ones((len(text),1)).cuda()

                self.run_step(latent_codes, cond, loss_L1)

                if (self.step % self.log_interval == 0) and ((self.args.distributed and is_main_process()==True) or not self.args.distributed):
                    info_dict = logger.get_current().name2val
                    for k,v in info_dict.items():
                        if k == 'loss':

                            log_str = 'step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v)
                            self.logger.info(log_str)
                            self.log_writer.add_scalar('Loss/loss', float(info_dict['loss']), self.step+self.resume_step)

                        if k in ['step', 'samples'] or '_q' in k:
                            continue

                if self.step % self.save_interval == 0:
                    if self.args.distributed:
                        if is_main_process()==True:
                            self.save_distributed()
                    else:
                        self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if is_main_process()==True:
                self.save()
            self.evaluate()

    def evaluate(self):

        return


    def run_step(self, batch, cond, loss_L1, loss_args=None):
        self.forward_backward(batch, cond, loss_L1, loss_args=loss_args)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, loss_L1, loss_args=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch 
            micro_cond = cond
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                loss_L1,
                model_kwargs=micro_cond,
                dataset=self.data.dataset,
                loss_args=loss_args
            )
            
            losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            weights = weights.to(losses["loss"].device)

            loss = losses["loss"]
            

            log_loss_dict(
                self.diffusion, t, {k: v for k, v in losses.items()}
            )
            
            self.mp_trainer.backward(loss)

    def _anneal_lr(self, gama=0.9):
        if self.step == 0:
            return
        if (self.step) % 1000 == 0:
            if self.lr <= 1e-7:
                return
            lr = self.lr * gama
            self.lr = lr
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            print(f'adjust_lr:{lr}')

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save_distributed(self):
        print("main thread saving state dict.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        state_dict = model_to_save.state_dict()

        def save_checkpoint_dis(state_dict):

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint_dis(state_dict)


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.item())
        