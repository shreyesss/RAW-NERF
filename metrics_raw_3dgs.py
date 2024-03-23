# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation script."""

import os 
import matplotlib.pyplot as plt 
import functools
from os import path
import sys
import time
import gin
from absl import app
from jax import random
from internal import configs
configs.define_common_flags()
# jax.config.parse_flags_with_absl()

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf 
from internal.gs_utils import ssim, psnr
from internal import train_utils
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

def readImages_RAW(renders_dir1, renders_dir2, gt_dir):
    renders1, renders2, renders3, renders_raw, = [], [], [], []
    gts, gts_raw = [], []
    image_names = []
 
    for fname in os.listdir(renders_dir1):
        # print(fname)
        render1 = Image.open(renders_dir1 / fname)
        render2 = Image.open(renders_dir2 / fname)
        gt = Image.open(gt_dir / fname)
       

        renders1.append(tf.to_tensor(render1).unsqueeze(0)[:, :3, :, :].cuda())
        renders2.append(tf.to_tensor(render2).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
       
      
        image_names.append(fname)
 
    return renders1, renders2,gts, image_names



def evaluate(scene_dir, is_raw, do_train):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

  
      
    print("Scene:", scene_dir)
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}
    print(scene_dir)
    test_dir = Path(scene_dir) / "test"



    full_dict[scene_dir] = {}
    per_view_dict[scene_dir]= {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

  
    gt_dir = test_dir/ "gt"
    renders_dir1 = test_dir / "renders"
    renders_dir2 = test_dir / "renders_cc"
        
      
    renders1, renders2, gts, image_names = readImages_RAW(renders_dir1,renders_dir2, gt_dir)

    ssims = []
    psnrs = []
    


    for idx in tqdm(range(len(renders1)), desc="Metric evaluation progress (postprocess)"):
        ssims.append(ssim(renders1[idx], gts[idx]))
        psnrs.append(psnr(renders1[idx], gts[idx]))
        # lpipss.append(lpips(renders1[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    # print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")
    full_dict[scene_dir].update({"SSIM1": torch.tensor(ssims).mean().item(),
                                            "PSNR1": torch.tensor(psnrs).mean().item()})
    
    per_view_dict[scene_dir].update({"PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}})
  
    ssims = []
    psnrs = []
    lpipss = []
    
    for idx in tqdm(range(len(renders2)), desc="Metric evaluation progress (affine color transform)"):
        ssims.append(ssim(renders2[idx], gts[idx]))
        psnrs.append(psnr(renders2[idx], gts[idx]))
        # lpipss.append(lpips(renders2[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
 
    print("")
    full_dict[scene_dir].update({"SSIM2": torch.tensor(ssims).mean().item(),
                                            "PSNR2": torch.tensor(psnrs).mean().item()})

    with open(scene_dir + "/results.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
              






def main(unused_argv):
  config = configs.load_config(save_config=False)
  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)
  print(config.checkpoint_dir)
  scene_dir = os.path.join(config.checkpoint_dir, "test_preds")
  evaluate(scene_dir, True, False)

if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
