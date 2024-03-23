
# import rawpy 
# import json 
# from internal import utils
# from PIL import Image 
# import numpy as np
# import jax
# import jax.numpy as jnp
# from internal.raw_utils import *
# import matplotlib.pyplot as plt
# import cv2 

# def processing_fn(x):
#         x_jax = jnp.array(x)
#         x_demosaic_jax = bilinear_demosaic_jax(x_jax)
#         # if n_downsample > 1:
#         #     x_demosaic_jax = lib_image.downsample(x_demosaic_jax, n_downsample)
#         return np.array(x_demosaic_jax)

# def load_raw_exif():
#     base = "/home/cilab/shreyas/codes/multinerf/data/rawnerf/scenes/bikes/raw/IMG_3353"
#     with utils.open_file(base + '.dng', 'rb') as f:
#       raw = rawpy.imread(f).raw_image
#     with utils.open_file(base + '.json', 'rb') as f:
#       exif = json.load(f)[0]
#     return raw, exif


# image = Image.open("/home/cilab/shreyas/codes/multinerf/data/rawnerf/scenes/bikes/images/IMG_3353.JPG")
# raw, _ = load_raw_exif()
# print(raw.min(), raw.max())
# raw = (raw -528) / (4095 -528)
# bgr = cv2.cvtColor(raw.astype(np.uint16), cv2.COLOR_BAYER_BG2RGB)  # The result is BGR format with 16 bits per pixel and 12 bits range [0, 2^12-1].
# plt.imshow(bgr)
# plt.show()
# # print(raw.min(), raw.max())
# # raw = processing_fn(raw)
# # print(raw.min(), raw.max())

# # plt.imshow(raw)
# # plt.show()



# # print(raw)
# print(raw.min(), raw.max())

# import rawpy
# from internal import utils
# import matplotlib.pyplot as plt

# path = "./data/rawnerf/scenes/bikes/raw/IMG_3353.dng"

# with utils.open_file(path, 'rb') as f:
#        raw = rawpy.imread(f).raw_image

# # with rawpy.imread(path) as raw:
# #     print(raw.min(), raw.max())
# rgb = raw.postprocess(use_camera_wb=True, use_auto_wb=False, output_color=rawpy.ColorSpace.sRGB)

# plt.imshow(rgb)
# plt.show()
# import numpy as np
# import jax.numpy as jnp
# from jax.scipy.signal import convolve
# import torch 
# import torch.nn.functional as F
# input_tensor = torch.ones(1, 3, 756, 1008)
# kernel = 2*torch.ones(3, 1, 11, 11)
# window_size = 11



import numpy as np
from scipy.signal import convolve2d

import numpy as np
from scipy.signal import convolve2d

def conv2d(input, kernel, padding='SAME'):
    if padding == 'SAME':
        pad_height = (kernel.shape[-2] - 1) // 2
        pad_width = (kernel.shape[-1] - 1) // 2
        # Ensure pad_total has shape (4, 2)
        pad_total = ((pad_height, pad_height), (pad_width, pad_width))
        input = np.pad(input, pad_total, mode='constant')
    # Perform convolution using scipy.signal.convolve2d
    return convolve2d(input, kernel, mode='same', boundary='fill', fillvalue=0)

# Example usage:
input_shape = (1, 3, 756, 1008)
kernel_shape = (3, 11, 11)



# Assuming input tensor shapes are torch.Size([1, 3, 756, 1008]) and kernel torch.Size([3, 1, 11, 11])
# Convert them to JAX arrays
input_tensor = np.ones([1, 3, 756, 1008])
kernel = 2*np.ones([3, 1, 11, 11])
window_size = 11
# input_jax = jnp.array(input_tensor)  # Assuming input_tensor is your PyTorch tensor
# kernel_jax = jnp.array(kernel)      # Assuming kernel is your PyTorch kernel tensor

# Perform convolution
output_jax = conv2d(input_tensor, kernel, padding='SAME')
# output_torch = F.conv2d(input_tensor, kernel, padding=window_size // 2, groups=3) 
print(output_jax)
# print(output_torch)
    

