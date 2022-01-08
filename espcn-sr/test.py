import argparse
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import ESPCN
import cloudpickle
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cloudpickle.load(open(args.net_file, "rb")).to(device)
    model.eval()

    args.image_file = osp.abspath(args.image_file)

    image = pil_image.open(args.image_file).convert("RGB")

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize(
        (hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC
    )
    bicubic = lr.resize(
        (lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC
    )
    bicubic.save(args.image_file.replace(".", "_bicubic_x{}.".format(args.scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print("PSNR (ESPCN): {:.2f}".format(psnr))
    print("PSNR (BICUBIC): {:.2f}".format(calc_psnr(hr, _)))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace(".", "_espcn_x{}.".format(args.scale)))
