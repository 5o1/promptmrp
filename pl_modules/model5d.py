import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib

import pytorch_lightning as pl

from data.blacklist import FileBoundBlacklist
from models.modules.context import ExtraContext

def resolve_class(class_path: str):
    """Dynamically resolve a class from its string path."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

class CascadesModule(MriModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        compute_sens_per_coil: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # two flags for reducing memory usage
        self.compute_sens_per_coil = compute_sens_per_coil

        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model = model

        self.lastloss = []
        self.loss = SSIMLoss()

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", compute_sens_per_coil=False):
        return self.model(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil=compute_sens_per_coil)   

    def training_step(self, batch, batch_idx):
        if not torch.isfinite(batch.masked_kspace).all():
            raise ValueError(f"Invalid masked_kspace in batch {batch_idx}")

        with ExtraContext(self) as ctx:
            output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type, compute_sens_per_coil=self.compute_sens_per_coil)
            output = output_dict['img_pred']
            target, output = transforms.center_crop_to_smallest(
                batch.target, output)

            loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
            assert len(ctx.losses) >0
            for lossname, localloss in ctx.losses:
                loss += localloss

        self.log("train_loss", loss, prog_bar=True)

        with torch.no_grad():
            if self.lastloss and loss - self.lastloss[-1] > 0.1:
                print(f"\nWarning: loss increased from {self.lastloss} to {loss.item()} at batch{batch_idx} fname {batch.fname} slice_num {batch.slice_num} max_value {batch.max_value}")
            self.lastloss.append(loss.item())
            if len(self.lastloss) > 10:
                self.lastloss.pop(0)
            ##! raise error if loss is nan
            if torch.isnan(loss):
                bad_index = f'{batch.fname}@{batch.slice_num}'
                if self.blacklist:
                    self.blacklist.append(bad_index)
                # raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
                warning_msg = f"Warning: Encountered NaN in batch {bad_index}. Skipping this batch."
                print(warning_msg)
                if self.logger and isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.experiment.log({"warning": warning_msg, "skipped_batch_idx": batch_idx})
                return None
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(
            batch.target, img_zf)
        val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        cc = batch.masked_kspace.shape[1]
        centered_coil_visual = torch.log(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_visual, 
            "sens_maps": output_dict['sens_maps'][:,0].abs(),
            "output": output,
            "target": target,
            "loss": val_loss,
            "dataloader_idx": dataloader_idx,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']

        crop_size = batch.crop_size 
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)

        num_slc = batch.num_slc
        return {
            'output': output.cpu(), 
            'slice_num': batch.slice_num, 
            'fname': batch.fname,
            'num_slc':  num_slc
        }
        