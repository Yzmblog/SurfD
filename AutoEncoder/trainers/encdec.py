from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from hesiod import get_out_dir, hcfg
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from AutoEncoder.data.dataset import UdfsDataset
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import compute_gradients, progress_bar, random_point_sampling
import numpy as np
import os
from encdec.DynamicSampler import WeightedDynamicSampler, DynamicBatchSampler, SequenceSampler, SequenceSampler_Train

import wandb

use_wandb = True
if use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="AutoEncoder",
        name="training",
        config={
        "learning_rate": 1e-4,
        "epochs": 20000,
        }
    )

class EncoderDecoderTrainer:
    def __init__(self) -> None:
        train_ids_file = Path(hcfg("dset.train_ids_file", str))
        dset_split = hcfg("dset.split", str)
        dset_root = Path(hcfg("dset.root", str))
        name = hcfg("dset.name", str)

        self.evaluation = False

        out_dir = Path(hcfg("log_dir", str))

        self.train_dset = UdfsDataset(name, dset_root, dset_split)
        train_bs = hcfg("train_bs", int)

        val_bs = hcfg("val_bs", int)
        
        if 'curriculum' in name:
            training_idxes = self.train_dset.get_training_idxes()
            print("first sample nums: ", len(training_idxes))
            weights = [1 if i in training_idxes else 0 for i in range(len(self.train_dset))]
        
            #self.sampler = WeightedDynamicSampler(weights, len(self.train_dset))
            self.sampler = SequenceSampler_Train(training_idxes)
            self.batch_sampler = DynamicBatchSampler(sampler = self.sampler, batch_size=train_bs, drop_last=False)
            self.train_loader = DataLoader(self.train_dset, batch_sampler=self.batch_sampler)

            self.val_sampler = SequenceSampler(training_idxes, len(self.train_dset))
            self.val_batch_sampler = DynamicBatchSampler(sampler = self.val_sampler, batch_size=val_bs, drop_last=False)
            self.val_loader = DataLoader(self.train_dset, batch_sampler=self.val_batch_sampler)

        else:
            self.train_loader = DataLoader(
                self.train_dset,
                train_bs,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

        self.num_points_pcd = hcfg("num_points_pcd", int)
        latent_size = hcfg("latent_size", int)
        self.max_dist = hcfg("udf_max_dist", float)
        self.num_points_forward = hcfg("num_points_forward", int)

        encoder = Dgcnn(latent_size)
        self.encoder = encoder.cuda()

        self.coords_encoder = CoordsEncoder()

        decoder_cfg = hcfg("decoder", Dict[str, Any])
        decoder = CbnDecoder(
            self.coords_encoder.out_dim,
            latent_size,
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers"],
        )
        self.decoder = decoder.cuda()

        params = list(encoder.parameters())
        params.extend(decoder.parameters())
        lr = hcfg("lr", float)
        self.optimizer = Adam(params, lr)

        self.epoch = 0
        self.global_step = 0

        self.ckpts_path = out_dir / "ckpts"

        tune = False
        if tune:
            self.restore_from_last_ckpt()

        else:
            if self.ckpts_path.exists():
                self.restore_from_last_ckpt()

        os.makedirs(str(self.ckpts_path), exist_ok=True)

        self.logger = SummaryWriter(out_dir / "logs")

    def train(self) -> None:
        name = hcfg("dset.name", str)
        num_epochs = hcfg("num_epochs", int)
        start_epoch = self.epoch
        best_val_loss = 1e9
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            epoch_losses = []
            epoch_loss = 0

            self.encoder.train()
            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                indexes, _, pcds, coords, gt_udf, gt_grad = batch
                pcds = pcds.cuda()
                coords = coords.cuda()
                gt_udf = gt_udf.cuda()
                gt_grad = gt_grad.cuda()


                pcds = random_point_sampling(pcds, self.num_points_pcd)

                gt_udf = gt_udf / self.max_dist
                gt_udf = 1 - gt_udf
                c_u_g = torch.cat([coords, gt_udf.unsqueeze(-1), gt_grad], dim=-1)

                selected_c_u_g = random_point_sampling(c_u_g, self.num_points_forward)
                selected_coords = selected_c_u_g[:, :, :3]
                selected_coords.requires_grad = True
                selected_gt_udf = selected_c_u_g[:, :, 3]
                selected_gt_grad = selected_c_u_g[:, :, 4:]

                latent_codes = self.encoder(pcds)

                coords_encoded = self.coords_encoder.encode(selected_coords)
                pred = self.decoder(coords_encoded, latent_codes)


                udf_loss = F.binary_cross_entropy_with_logits(pred, selected_gt_udf)

                udf_pred = torch.sigmoid(pred)
                

                udf_pred = 1 - udf_pred
                udf_pred = udf_pred * self.max_dist

                gradients = compute_gradients(selected_coords, udf_pred)

                grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")


                mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
                grad_loss = grad_loss[mask].mean()


                self.optimizer.zero_grad()

                loss = udf_loss + 0.1*grad_loss

                epoch_losses.append(udf_loss.detach().cpu().item())

                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logger.add_scalar(
                        "train/udf_loss",
                        udf_loss.item(),
                        self.global_step,
                    )
                    self.logger.add_scalar(
                        "train/grad_loss",
                        grad_loss.item(),
                        self.global_step,
                    )
                    if use_wandb:
                        wandb.log({"train/udf_loss": udf_loss.item(), "train/grad_loss": grad_loss.item()}, step=self.global_step) #"train/grad_loss": grad_loss.item()}, step=self.global_step)

                    print(f'steps: {self.global_step}; loss: {loss.item()}; udf_loss: {udf_loss.item()}; grad_loss: {grad_loss.item()}') #eikonal_loss: {gradient_error.item()}

                self.global_step += 1

            epoch_loss = np.mean(epoch_losses)
            print(f'epoch_udf_loss: {epoch_loss}')

            switch_epoch = 64
            curr_training_idxes = self.train_dset.get_training_idxes().copy()
            total_samples = len(self.train_dset)
            if epoch % switch_epoch == 63 and 'curriculum' in name and len(curr_training_idxes) < total_samples:

                _, _, _, new_add_idxes = self.val()

                self.train_dset.update_training_idxes(new_add_idxes)

                #self.batch_sampler.update_weights(new_weights)
                self.batch_sampler.update_training_idxes(new_add_idxes)
                self.val_batch_sampler.update_training_idxes(new_add_idxes)
                print(f'training samples now: {len(curr_training_idxes)+len(new_add_idxes)}')

                assert(len(set(curr_training_idxes)&set(new_add_idxes)) == 0)
                print(curr_training_idxes, new_add_idxes)

            if epoch % 1000 == 0:
                self.save_ckpt(all=True)

            self.save_ckpt()
        if use_wandb:
            wandb.finish()

    def val(self) -> float:
        self.encoder.eval()
        self.decoder.eval()
        print('evaluation now')

        val_losses = []
        udf_losses = []
        grad_losses = []
        remain_indexes = []

        for batch in progress_bar(self.val_loader):
            indexes, _, pcds, coords, gt_udf, gt_grad = batch
            pcds = pcds.cuda()
            coords = coords.cuda()
            gt_udf = gt_udf.cuda()
            gt_grad = gt_grad.cuda()
            indexes = list(indexes.cpu().numpy())

            pcds = random_point_sampling(pcds, self.num_points_pcd)

            gt_udf = gt_udf / self.max_dist
            gt_udf = 1 - gt_udf
            c_u_g = torch.cat([coords, gt_udf.unsqueeze(-1), gt_grad], dim=-1)

            selected_c_u_g = random_point_sampling(c_u_g, self.num_points_forward)
            selected_coords = selected_c_u_g[:, :, :3]
            selected_coords.requires_grad = True
            selected_gt_udf = selected_c_u_g[:, :, 3]
            selected_gt_grad = selected_c_u_g[:, :, 4:]

            with torch.no_grad():
                latent_codes = self.encoder(pcds)

            selected_coords.requires_grad = True
            coords_encoded = self.coords_encoder.encode(selected_coords)
            pred = self.decoder(coords_encoded, latent_codes)

            udf_loss = F.binary_cross_entropy_with_logits(pred, selected_gt_udf)


            udf_pred = torch.sigmoid(pred)
            udf_pred = 1 - udf_pred
            udf_pred *= self.max_dist

            udf_pred.sum().backward()
            gradients = selected_coords.grad
            
            selected_coords.grad.zero_()
            grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")
            
            mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
            grad_loss = grad_loss[mask].mean()

            loss = udf_loss + 0.1*grad_loss

            val_losses.append(loss.detach().cpu().item())
            udf_losses.append(udf_loss.detach().cpu().item())
            grad_losses.append(grad_loss.detach().cpu().item())
            remain_indexes.append(indexes)
        

        val_losses = np.array(val_losses)
        val_losses = val_losses.reshape(-1)
        print('val_losses: ', val_losses)
        remain_indexes = np.array(remain_indexes).reshape(-1)
        new_can_idxes = list(np.argsort(val_losses)[:100])

        new_add_idxes = [remain_indexes[i] for i in new_can_idxes]
        return np.mean(val_losses), np.mean(udf_losses), np.mean(grad_losses), new_add_idxes


    def save_ckpt(self, all: bool = False, best=False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if best:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "best" in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()
            ckpt_path = self.ckpts_path / f"best_{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        elif all:
            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)
        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "last" in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"last_{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:

        if self.ckpts_path.exists():

            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "last" in p.name]
            error_msg = "Expected only one last ckpt, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
