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

from absl import app
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.signal import convolve
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()

# metric helper functions 

# def psnr(img1, img2):
#     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * np.log10(1.0 / np.sqrt(mse))



# # torch.Size([1, 3, 756, 1008]) torch.Size([3, 1, 11, 11])
# def gaussian(window_size, sigma):
#     x = jnp.arange(window_size)
#     gauss = jnp.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
#     return gauss / jnp.sum(gauss)

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5)[:, jnp.newaxis]
#     _2D_window = jnp.dot(_1D_window, _1D_window.T).astype(jnp.float32)[jnp.newaxis, jnp.newaxis, :, :]
#     window = jnp.repeat(_2D_window, channel, axis=0)
#     return window

# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.shape[-3]
#     window = create_window(window_size, channel)
#     print(img1.shape, window.shape)

#     return _ssim(img1, img2, window, window_size, channel, size_average)

# def conv2d_with_padding(input, kernel, padding):
#     input_padded = jnp.pad(input, [(0, 0)] * (input.ndim - 2) + [(padding, padding), (padding, padding)], mode='constant')
#     return convolve(input_padded, kernel, mode='valid')

# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = conv2d_with_padding(img1, window, padding=window_size // 2)
#     mu2 = conv2d_with_padding(img2, window, padding=window_size // 2)

#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
    
#     sigma1_sq = conv2d_with_padding(img1 * img1, window, padding=window_size // 2) - mu1_sq
#     sigma2_sq = conv2d_with_padding(img2 * img2, window, padding=window_size // 2) - mu2_sq
#     sigma12 = conv2d_with_padding(img1 * img2, window, padding=window_size // 2) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return jnp.mean(ssim_map)
#     else:
#         return jnp.mean(jnp.mean(jnp.mean(ssim_map, axis=-1), axis=-1), axis=-1)



def write_metrics_to_file(values, fname):
    with open(fname, "w") as file:
        # Write the list of values to the file
        file.write("List of values:\n")
        for value in values:
            file.write(str(value) + "\n")

        # Calculate and write the mean
        mean_value = sum(values) / len(values)
        file.write("\nMean value: " + str(mean_value))





def main(unused_argv):
  config = configs.load_config(save_config=False)
  key = random.PRNGKey(20200823)
  print(key)
  import sys
  sys.exit()
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)
  cc_fun = image.color_correct

  print("Reading test dataset")
  test_dataset = datasets.load_dataset('test', config.data_dir, config)
  postprocess_fn = test_dataset.metadata['postprocess_fn']
  # train_dataset = datasets.load_dataset('train', config.data_dir, config)

  
  out_dir = path.join(config.checkpoint_dir,
                      'path_renders' if config.render_path else 'test_preds')
  
  # render_path_affine = os.path.join(out_dir), name, "ours_{}".format(iteration), "renders_affine")
  render_path_color_correct = os.path.join(out_dir,"test", "renders_cc")
  render_path = os.path.join(out_dir, "test","renders")
  gts_path = os.path.join(out_dir, "test", "gt")
  os.makedirs(render_path, exist_ok=True)
  os.makedirs(render_path_color_correct, exist_ok=True)
  os.makedirs(gts_path, exist_ok=True)


  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  step = int(state.step)
  print(f'Rendering checkpoint at step {step}.')
  if config.eval_save_output and (not utils.isdir(out_dir)):
    utils.makedirs(out_dir)

 
  key = random.PRNGKey(0 if config.deterministic_showcase else step)

  

  
  render_times = []
  for idx in range(test_dataset.size):
      eval_start_time = time.time()
      batch = next(test_dataset)
     
    
      print(f'Rendering image {idx+1}/{test_dataset.size}')
      rays = batch.rays
      train_frac = state.step / config.max_steps
      rendering = models.render_image(
          functools.partial(
              render_eval_pfn,
              state.params,
              train_frac,
          ),
          rays,
          None,
          config,
      )

      if jax.host_id() != 0:  # Only record via host 0.
        continue

      render_times.append((time.time() - eval_start_time))
      print(f'Rendered in {render_times[-1]:0.3f}s')

      # Cast to 64-bit to ensure high precision for color correction function.
      gt_rgb = np.array(batch.rgb_jpeg, dtype=np.float64)
      print(gt_rgb.shape, gt_rgb.min(), gt_rgb.max())
      gt_raw = np.array(batch.rgb, dtype=np.float64)
      plt.imsave(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"),(255*gt_rgb).astype(np.uint8))
      rendering_raw = np.array(rendering['rgb'], dtype=np.float64)
      # rendering_raw = (rendering_raw - rendering_raw.min()) / (rendering_raw.max() - rendering_raw.min())
     
  
      rendering_rgb = postprocess_fn(rendering_raw)
      print(rendering_rgb.shape, rendering_rgb.min(), rendering_rgb.max())
      plt.imsave(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"),(255*rendering_rgb).astype(np.uint8))

      rendering_cc = cc_fun(rendering_rgb, gt_rgb)
      print(rendering_cc.min(), rendering_cc.max())
      plt.imsave(os.path.join(render_path_color_correct, '{0:05d}'.format(idx) + ".png"),(255*rendering_cc).astype(np.uint8))
          

      
        # proces image for metric calculation 
    #     render = np.expand_dims(np.transpose(rendering_rgb_cc, (2,0,1)), 0)
    #     gt = np.expand_dims(np.transpose(gt_rgb, (2,0,1)), 0)
    #     print(render.shape)
    #     print(gt.shape)
    #     ssims.append(ssim(render, gt))
    #     psnrs.append(psnr(render, gt))

    # write_metrics_to_file(ssims, os.path.join(out_dir,"test", "ssims.txt"))
    # write_metrics_to_file(psnrs, os.path.join(out_dir,"test", "psnrs.txt"))


    
       

     



if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
