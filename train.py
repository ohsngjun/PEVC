import argparse
import math
import random
import shutil
import sys
import os

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ms_ssim

from dataset import *
from utils import net_aux_optimizer
from loramodel import ScaleSpaceFlowLora
from weight_entropy_module import *

import compressai.zoo.video as ssf


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target):
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames


        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)

            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()

        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        lambdas = torch.full_like(bpp_loss, self.lmbda)

        bpp_loss = bpp_loss.mean()
        out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss

        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True:
            aux_loss.requires_grad_(True)
            aux_loss.backward()

    return aux_loss_sum

def psnr(image1, image2, max_value=1):
    mse = torch.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_value / math.sqrt(mse))

def psnr_batch(batch1, batch2):
    total_psnr = 0
    imgs = 0
    
    for sublist1, sublist2 in zip(batch1, batch2):
        for img1, img2 in zip(sublist1, sublist2):
            total_psnr += psnr(img1, img2)
            imgs +=1
    
    return total_psnr / imgs

def ssim_batch(batch1, batch2):
    total_ssim = 0
    imgs = 0
    for sub1, sub2 in zip(batch1, batch2):
        total_ssim += ms_ssim(sub1, sub2, data_range=1)
        imgs += 1

    return total_ssim.item() / imgs


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }

    optimizer = net_aux_optimizer(args, net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, total_frames,
):
    model.train()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    test_psnr = []
    test_ssim = []

    spike_and_slap_cdf = SpikeAndSlabCDF()
    weight_entropyModule = WeightEntropyModule(spike_and_slap_cdf)
    weight_entropyModule.to(device)
    weight_entropyModule.train()

    for i, batch in enumerate(train_dataloader):
        d = [frames.to(device) for frames in batch]
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        _, _, H, W = d[0].size()

        out_net = model(d)

        test_psnr.append(psnr_batch(d, out_net['x_hat']))
        test_ssim.append(ssim_batch(d, out_net['x_hat']))

        w_bpp_sum = 0
        
        for key in model.state_dict():
            if 'lora' in key:  
                weight = model.state_dict()[key] 
                og_shape = weight.shape
                weight = weight.reshape(1,1,-1)

                w_hat, w_likelihood = weight_entropyModule(weight, True)
                w_hat = w_hat.reshape(og_shape)

                w_bpp = torch.log(w_likelihood) / (-math.log(2) * H * W * total_frames)
                w_bpp_sum += w_bpp.sum()


        out_criterion = criterion(out_net, d)
        total_loss = out_criterion["loss"] + w_bpp_sum
        total_loss.requires_grad_(True)
        total_loss.backward()
        loss.update(total_loss)
        bpp_loss.update(out_criterion["bpp_loss"])


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        aux_optimizer.step()

        if i % 25 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i}/{len(train_dataloader)}" 
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f}'
            )
    return loss.avg, round(sum(test_psnr)/len(test_psnr),4), round(sum(test_ssim)/len(test_ssim),4), bpp_loss.avg + w_bpp_sum


def test(args, test_dataloader, model, total_frames):
    spike_and_slap_cdf = SpikeAndSlabCDF()
    weight_entropyModule = WeightEntropyModule(spike_and_slap_cdf)
    weight_entropyModule.eval()

    device = next(model.parameters()).device

    weight_entropyModule.to(device)

    w_bpp_sum = 0
    state_dict = ssf.ssf2020(args.quality,pretrained=True).state_dict()
    
    for key in model.state_dict():
        if 'lora' in key:  
            weight = model.state_dict()[key] 
            og_shape = weight.shape
            weight = weight.reshape(1,1,-1)

            bits = weight_entropyModule.compress(weight)
            w_hat = weight_entropyModule.decompress(bits, og_shape)
            w_hat = w_hat.reshape(og_shape)

            w_bpp_sum += len(bits[0])*8/(args.patch_size[0]*args.patch_size[1])
            state_dict[key] = w_hat.cpu()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_psnr = []
    test_ssim = []
    bpp = []

    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            strings, shape, bits = model.compress(d)
            _, _, H, W = d[0].size()
            num_frames = len(d)
            num_pixels = H * W * num_frames
            
            bpp.append(bits/num_pixels)
            result = model.decompress(strings, shape)

            test_psnr.append(psnr_batch(d, result))
            test_ssim.append(ssim_batch(d, result))
        result = round(sum(test_psnr)/len(test_psnr),4), round(sum(test_ssim)/len(test_ssim),4), round(sum(bpp)/len(bpp)+ w_bpp_sum/total_frames,5), round(w_bpp_sum*H*W)

    return result



def save_checkpoint(state, is_best, path="./", filename="checkpoint.pth.tar"):
    os.makedirs(path, exist_ok=True)
    route = os.path.join(path,filename)
    torch.save(state, route)
    if is_best:
        best_route = os.path.join(path, 'best')
        os.makedirs(best_route, exist_ok=True)
        bestfilename = filename.rsplit('.')[0]
        shutil.copyfile(route, os.path.join(best_route, f"{bestfilename}_best.pth.tar"))



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--method",
        default='lora',
        choices=['lora', 'repeat'], 
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=15,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s), not important",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(1024, 1920),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")

    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--quality", type=int, default=1, help="Select bitrate")
    parser.add_argument("--train_gop", type=int, default=4)
    parser.add_argument("--save_name", type=str, default="test")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = VideoDataset(
        args.dataset,
        transform=train_transforms,
        frame_size=args.train_gop,
    )
    total_frames = train_dataset.__len__()*args.train_gop

    test_dataset = VideoDataset(
        args.dataset,
        transform=train_transforms,
        frame_size=12,
    )
    n_test_frames = test_dataset.__len__()*12

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ScaleSpaceFlowLora(args.method)
    net = net.to(device)
    
    base_ssf = ssf.ssf2020(args.quality,pretrained=True)
    net.load_state_dict(base_ssf.state_dict(), strict=False)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",patience=1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, return_details=True)
    

    last_epoch = 0
    if args.checkpoint: 
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        loss, psnr, ssim, bpp = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            total_frames,
        )

        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                path = f"checkpoints/{args.save_name}/{args.method}/quality{args.quality}",
                filename = f"ep{epoch}.pth.tar"
            )
        if epoch == args.epochs-1:
            result = test(args, test_dataloader, net, n_test_frames)
            print('result')
            print('PSNR: ',result[0],"  MS-SSIM: ",result[1], " bpp: ",result[2])
            

if __name__ == "__main__":
    main(sys.argv[1:])