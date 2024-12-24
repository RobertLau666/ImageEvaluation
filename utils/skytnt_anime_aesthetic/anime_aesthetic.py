import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional


# ========= Model =========

# copy from https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int[]): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.0,
            head_init_scale=1.0,
    ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_cfgs = {
    "atto": [[2, 2, 6, 2], [40, 80, 160, 320]],
    "femto": [[2, 2, 6, 2], [48, 96, 192, 384]],
    "pico": [[2, 2, 6, 2], [64, 128, 256, 512]],
    "nano": [[2, 2, 8, 2], [80, 160, 320, 640]],
    "tiny": [[3, 3, 9, 3], [96, 192, 384, 768]],
    "base": [[3, 3, 27, 3], [128, 256, 512, 1024]],
    "large": [[3, 3, 27, 3], [192, 384, 768, 1536]],
    "huge": [[3, 3, 27, 3], [352, 704, 1408, 2816]],
}


def convnextv2(cfg_name, **kwargs):
    cfg = model_cfgs[cfg_name]
    model = ConvNeXtV2(depths=cfg[0], dims=cfg[1], **kwargs)
    return model


# ========= Dataset =========

EXTENSION = [".png", ".jpg", ".jpeg"]


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def rescale_pad(image, output_size, random_pad=False):
    h, w = image.shape[-2:]
    if h != output_size or w != output_size:
        r = min(output_size / h, output_size / w)
        new_h, new_w = int(h * r), int(w * r)
        if random_pad:
            r2 = random.uniform(0.9, 1)
            new_h, new_w = int(new_h * r2), int(new_w * r2)
        ph = output_size - new_h
        pw = output_size - new_w
        left = random.randint(0, pw) if random_pad else pw // 2
        right = pw - left
        top = random.randint(0, ph) if random_pad else ph // 2
        bottom = ph - top
        image = transforms.functional.resize(image, [new_h, new_w])
        image = transforms.functional.pad(
            image, [left, top, right, bottom], random.uniform(0, 1) if random_pad else 0
        )
    return image


def random_crop(image, min_rate=0.8):
    h, w = image.shape[-2:]
    new_h, new_w = int(h * random.uniform(min_rate, 1)), int(w * random.uniform(min_rate, 1))
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    image = image[:, top: top + new_h, left: left + new_w]
    return image


class AnimeAestheticDataset(Dataset):
    def __init__(self, path, img_size, xflip=True):
        all_files = {
            os.path.relpath(os.path.join(root, fname), path)
            for root, _dirs, files in os.walk(path)
            for fname in files
        }
        all_images = sorted(
            fname for fname in all_files if file_ext(fname) in EXTENSION
        )
        with open(os.path.join(path, "label.json"), "r", encoding="utf8") as f:
            labels = json.load(f)
        image_list = []
        label_list = []
        for fname in all_images:
            if fname not in labels:
                continue
            image_list.append(fname)
            label_list.append(labels[fname])
        self.path = path
        self.img_size = img_size
        self.xflip = xflip
        self.image_list = image_list
        self.label_list = label_list

    def __len__(self):
        length = len(self.image_list)
        if self.xflip:
            length *= 2
        return length

    def __getitem__(self, index):
        real_len = len(self.image_list)
        fname = self.image_list[index % real_len]
        label = self.label_list[index % real_len]
        image = Image.open(os.path.join(self.path, fname)).convert("RGB")
        image = transforms.functional.to_tensor(image)
        image = random_crop(image, 0.8)
        image = rescale_pad(image, self.img_size, True)
        if index // real_len != 0:
            image = transforms.functional.hflip(image)
        label = torch.tensor([label], dtype=torch.float32)
        return image, label


# ========= Train =========


class AnimeAesthetic(pl.LightningModule):
    def __init__(self, cfg: str, drop_path_rate=0.0, ema_decay=0):
        super().__init__()
        self.net = convnextv2(cfg, in_chans=3, num_classes=1, drop_path_rate=drop_path_rate)
        self.ema_decay = ema_decay
        self.ema = None
        if ema_decay > 0:
            self.ema = deepcopy(self.net)
            self.ema.requires_grad_(False)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.net.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
        return optimizer

    def forward(self, x, use_ema=False):
        x = (x - 0.5) / 0.5
        net = self.ema if use_ema else self.net
        return net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = F.mse_loss(self.forward(images, False), labels)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        mae = F.l1_loss(self.forward(images, False), labels)
        logs = {"val/mae": mae}
        if self.ema is not None:
            mae_ema = F.l1_loss(self.forward(images, True), labels)
            logs["val/mae_ema"] = mae_ema
        self.log_dict(logs, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            with torch.no_grad():
                for ema_v, model_v in zip(
                        self.ema.state_dict().values(), self.net.state_dict().values()
                ):
                    ema_v.copy_(
                        self.ema_decay * ema_v + (1.0 - self.ema_decay) * model_v
                    )


def main(opt):
    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
    torch.manual_seed(0)
    np.random.seed(0)
    print("---load dataset---")
    full_dataset = AnimeAestheticDataset(opt.data, opt.img_size)
    full_dataset_len = len(full_dataset)
    train_dataset_len = int(full_dataset_len * opt.data_split)
    val_dataset_len = full_dataset_len - train_dataset_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_dataset_len, val_dataset_len]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size_train,
        shuffle=True,
        persistent_workers=True,
        num_workers=opt.workers_train,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size_val,
        shuffle=False,
        persistent_workers=True,
        num_workers=opt.workers_val,
        pin_memory=True,
    )
    print(f"train: {len(train_dataset)}")
    print(f"val: {len(val_dataset)}")
    print("---define model---")
    if opt.resume != "":
        anime_aesthetic = AnimeAesthetic.load_from_checkpoint(
            opt.resume, cfg=opt.cfg, drop_path_rate=opt.drop_path, ema_decay=opt.ema_decay
        )
    else:
        anime_aesthetic = AnimeAesthetic(cfg=opt.cfg, drop_path_rate=opt.drop_path, ema_decay=opt.ema_decay)

    print("---start train---")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch={epoch},mae={val/mae:.4f}",
    )
    callbacks = [checkpoint_callback]
    if opt.ema_decay > 0:
        checkpoint_ema_callback = ModelCheckpoint(
            monitor="val/mae_ema",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
            filename="epoch={epoch},mae-ema={val/mae_ema:.4f}",
        )
        callbacks.append(checkpoint_ema_callback)
    trainer = Trainer(
        precision=32 if opt.fp32 else 16,
        accelerator=opt.accelerator,
        devices=opt.devices,
        max_epochs=opt.epoch,
        benchmark=opt.benchmark,
        accumulate_grad_batches=opt.acc_step,
        val_check_interval=opt.val_epoch,
        log_every_n_steps=opt.log_step,
        strategy="ddp_find_unused_parameters_false" if opt.devices > 1 else None,
        callbacks=callbacks,
    )
    trainer.fit(anime_aesthetic, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--cfg",
        type=str,
        default="tiny",
        choices=list(model_cfgs.keys()),
        help="model configure",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="resume training from ckpt"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=768,
        help="image size for training and validation",
    )

    # dataset args
    parser.add_argument(
        "--data", type=str, default="./data", help="dataset path"
    )
    parser.add_argument(
        "--data-split",
        type=float,
        default=0.9999,
        help="split rate for training and validation",
    )

    # training args
    parser.add_argument("--epoch", type=int, default=100, help="epoch num")
    parser.add_argument(
        "--batch-size-train", type=int, default=16, help="batch size for training"
    )
    parser.add_argument(
        "--batch-size-val", type=int, default=2, help="batch size for val"
    )
    parser.add_argument(
        "--workers-train",
        type=int,
        default=4,
        help="workers num for training dataloader",
    )
    parser.add_argument(
        "--workers-val",
        type=int,
        default=4,
        help="workers num for validation dataloader",
    )
    parser.add_argument(
        "--acc-step", type=int, default=8, help="gradient accumulation step"
    )
    parser.add_argument(
        "--drop-path", type=float, default=0.1, help="Drop path rate"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.9999, help="use ema if ema-decay > 0"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"],
        help="accelerator",
    )
    parser.add_argument("--devices", type=int, default=4, help="devices num")
    parser.add_argument(
        "--fp32", action="store_true", default=False, help="disable mix precision"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=True, help="enable cudnn benchmark"
    )
    parser.add_argument(
        "--log-step", type=int, default=2, help="log training loss every n steps"
    )
    parser.add_argument(
        "--val-epoch", type=int, default=0.025, help="valid and save every n epoch"
    )

    opt = parser.parse_args()
    print(opt)

    main(opt)
