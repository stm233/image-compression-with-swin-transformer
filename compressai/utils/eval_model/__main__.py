# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision

import compressai

from compressai.zoo import load_state_dict, models

from compressai.models.retinanet.dataloader import CocoDataset, Resizer, Normalizer

from pycocotools.cocoeval import COCOeval
import gc

# from ptflops import get_model_complexity_info
from fvcore.nn import flop_count_table
from fvcore.nn import FlopCountAnalysis

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def reconstruct(reconstruction, filename, recon_path):
    reconstruction = reconstruction.squeeze()
    reconstruction.clamp_(0, 1)
    reconstruction = transforms.ToPILImage()(reconstruction.cpu())
    reconstruction.save(os.path.join(recon_path, filename))


@torch.no_grad()
def inference(model, x, filename, recon_path):
    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()

    out_enc = model.compress(x_padded)

    enc_time = time.time() - start
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    reconstruct(out_dec["x_hat"], filename, recon_path)         # add

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        # "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, context, filename, recon_path):
    input_image = x['img']
    input_image = input_image.unsqueeze(0)
    # begin = 100
    # size = 640
    # x = x[:,:,begin:begin+size,begin:begin+size]
    num_pixels = input_image.size(0) * input_image.size(2) * input_image.size(3)

    # x = x.unsqueeze(0)
    # print(x.shape)
    h, w = input_image.size(2), input_image.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        input_image,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    # print(filename,x.shape,x_padded.shape)
    start = time.time()
    # img_size = 256
    # b = x.unfold(2,img_size,img_size).unfold(3,img_size,img_size)
    # c = b.permute(0,2,3,1,4,5).contiguous().view(-1,3,img_size,img_size)
    # out_net = model.forward(c)
    # grid_img = torchvision.utils.make_grid(out_net["x_hat"], nrow=x.size(3)//img_size,padding=0)
    # grid_img = grid_img.unsqueeze(0)
    # print(grid_img.shape,x.shape)
    if context == None:
        out_net = model.forward(x_padded)
    else:
        context = context.unsqueeze(0)
        # context = context[:,:,begin:begin+size,begin:begin+size]
        h_c, w_c = context.size(2), context.size(3)
        p = 64  # maximum 6 strides of 2
        distance_h = h - h_c
        distance_w = w- w_c

        # new_h_c = (h_c + p - 1) // p * p
        # new_w_c = (w_c + p - 1) // p * p
        padding_left_c = padding_left + distance_w // 2
        padding_right_c = padding_right + (distance_w - distance_w // 2)
        padding_top_c = padding_top +  distance_h // 2
        padding_bottom_c = padding_bottom + (distance_h - distance_h // 2)
        context = F.pad(
                    context,
                    (padding_left_c, padding_right_c, padding_top_c, padding_bottom_c),
                    mode="constant",
                    value=0,
                )
        # print(x_padded.shape,context.shape)
        out_net = model.forward(x_padded,context)

    scale = x['scale']
    scores, labels, boxes = out_net["scores"], out_net["labels"], out_net["boxes"]
    boxes /= scale




    # grid_img = out_net["x_hat"]
    # grid_img = F.pad(
    #     grid_img, (-padding_left, -padding_right, -padding_top, -padding_bottom)
    # )
    
    elapsed_time = time.time() - start

    
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    # ms_ssim(x, grid_img, data_range=1.0)
    print(filename,"bpp", bpp.item())
    # print('num_pixels',num_pixels)
    # for likelihoods in out_net["likelihoods"].values():
    #     tmpBPP = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    #     print(tmpBPP)

    # reconstruct(grid_img, filename, recon_path)

    return {
        # "psnr": psnr(x, grid_img), # out_net["x_hat"]
        # "ms-ssim": ms_ssim(x, grid_img, data_range=1.0).item(),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()
    # return models[arch]().eval()


def eval_model(model, filepaths, entropy_estimation=True, half=False, recon_path='reconstruction'):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    # start collecting results
    results = []
    image_ids = []
    bpps = 0
    pixels = 0
    with torch.no_grad():
        for index in range(1,5000): # len(filepaths)

            x = filepaths[index]
            input_image = x['img'].to(device)
            input_image = input_image.unsqueeze(0)
            input_image = input_image.permute(0, 3, 1, 2).contiguous()
            num_pixels = input_image.size(0) * input_image.size(2) * input_image.size(3)

            # macs, params = get_model_complexity_info(model, (3, input_image.size(2), input_image.size(3)), as_strings=True,
            #                                print_per_layer_stat=True, verbose=True)

            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # flops = FlopCountAnalysis(model, input_image)
            # print(flop_count_table(flops))
            # break
            # start = time.time()
            out_net = model.forward(input_image)
            # elapsed_time = time.time() - start
            # print("encoding_time",elapsed_time)
            # break
            scale = x['scale']
            scores, labels, boxes = out_net["scores"], out_net["labels"], out_net["boxes"]
            boxes /= scale
            bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in out_net["likelihoods"].values()
                )
            # bpp = 0
            bpps += float(bpp)
            pixels += int(num_pixels)
            print(index,'/',len(filepaths), 'bpp = ',bpp,'pixels = ',num_pixels)
            # print('{}/{}'.format(index, len(filepaths)), end='\r')

            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < 0.05:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': filepaths.image_ids[index],
                        'category_id': filepaths.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    print(index, '/', len(filepaths), 'bpp = ', float(bpp), 'pixels = ', num_pixels,'category_id',
                          filepaths.label_to_coco_label(label), 'score = ',float(score),'bbox', box.tolist() )
                    # append detection to results
                    results.append(image_result)
                    # json.dump(image_result, open('{}_bbox_results.json'.format(filepaths.set_name), 'a+'), indent=4)

                    # append image to list of processed images
            image_ids.append(filepaths.image_ids[index])

            # del boxes
            # del labels
            # del scores
            # del bpp
            # del input_image
            # del x
            # gc.collect()
            # torch.cuda.empty_cache()
    print('average bpp',bpps/len(filepaths),'average pixels = ',pixels/len(filepaths))
    # write output
    json.dump(results, open('{}_bbox_results.json'.format(filepaths.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = filepaths.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(filepaths.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

        # x = read_image(f).to(device)
        # context = None

        # context_path = f.split("/")
        # context_path[-2] = 'L_GAN_x4_decompress'
        # context_path = os.path.join(*context_path)
        # # print(f)
        # # print(context_path)
        # context = read_image('/'+context_path).to(device)

    #     if not entropy_estimation:
    #         if half:
    #             model = model.half()
    #             x = x.half()
    #         rv = inference(model, x, _filename, recon_path)
    #     else:
    #         rv = inference_entropy_estimation(model, x,context , _filename, recon_path)
    #     for k, v in rv.items():
    #         metrics[k] += v
    # for k, v in metrics.items():
    #     metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser()

    # Common options.
    # original : /media/tianma/0403b42c-caba-4ab7-a362-c335a178175e/supervised-compression-main/dataset/coco2017/
    # BGP : /media/tianma/0403b42c-caba-4ab7-a362-c335a178175e/BPG_val2017/decompress/qp41
    # VTM :  /media/tianma/0403b42c-caba-4ab7-a362-c335a178175e/val2017/decompress
    parent_parser.add_argument("-d", "--dataset",default='/media/tianma/0403b42c-caba-4ab7-a362-c335a178175e/supervised-compression-main/dataset/coco2017/', type=str, help="dataset path")
    parent_parser.add_argument("-r", "--recon_path", type=str, default="reconstruction", help="where to save recon img")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        default='cnn2',
        type=str,
        choices=models.keys(),
        help="model architecture",
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        default=True,
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        default=False,
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        default=True,
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
            "-p",
            "--path",
            default='/home/tianma/Documents/STF-main/cnn2_100/98.ckpt',
            dest="paths",
            type=str,
            nargs="*",
            help="checkpoint path",
        )
    return parent_parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = CocoDataset(args.dataset, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    # filepaths = collect_images(args.dataset)
    # if len(filepaths) == 0:
    #     print("Error: no images found in directory.", file=sys.stderr)
    #     sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = args.paths
    opts = (args.architecture,)
    load_func = load_checkpoint
    log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    model = load_func(*opts, runs)
    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda:0")

    model.update(force=True)

    metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.recon_path)
    # for run in runs:
    #     if args.verbose:
    #         sys.stderr.write(log_fmt.format(*opts, run=run))
    #         sys.stderr.flush()
    #     model = load_func(*opts, run)
    #     if args.cuda and torch.cuda.is_available():
    #         model = model.to("cuda:0")
    #
    #     model.update(force=True)
    #
    #     metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.recon_path)
        # for k, v in metrics.items():
        #     results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
