import os
from pathlib import Path
import streamlit as st
from inference.interact.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
    overlay_davis
)
from inference.interact.interactive_utils import overlay_davis_torch
from utils import pal, create_dir
import torch
import numpy as np
from PIL import Image
from model.network import XMem
from inference.inference_core import InferenceCore

# constants
pal = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'

# utilities
def create_dir(dir):
    os.makedirs(dir, exist_ok=True)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}

def save_mask(dir:str, mask: np.ndarray):
    color_map_np = np.frombuffer(pal, dtype=np.uint8).reshape(-1, 3).copy()
    color_map_np = (color_map_np.astype(np.float32)*1.5).clip(0, 255).astype(np.uint8)
    colored_mask = color_map_np[mask]
    Image.fromarray(colored_mask).save(dir)

def save_overlay(dir:str, overlay: Image):
    Image.fromarray(overlay).save(dir)  
  
def get_mask_from_video_stream(cap, base_dir, mask):
  torch.set_grad_enabled(False)
 
  network = XMem(config, "./saves/XMem.pth").eval().to(device)
      
  processor = InferenceCore(network, config=config)
  processor.set_all_labels(range(1, 2)) # consecutive labels
  
  current_frame_index = 0
  base_mask_path = os.path.join(base_dir, 'masks')
  base_overlay_path = os.path.join(base_dir, 'overlay')
  create_dir(base_mask_path)
  create_dir(base_overlay_path)
  with torch.cuda.amp.autocast(enabled=True):
    while (cap.isOpened()):
      # load frame-by-frame
      _, frame = cap.read()
      if frame is None:
        break

      # convert numpy array to pytorch tensor format
      frame_torch, _ = image_to_torch(frame, device=device)
      print('frame_torch shape: ', frame_torch.shape)
      print('mask shape: ', mask.shape)
      if current_frame_index == 0:
        # initialize with the mask
        mask_torch = index_numpy_to_one_hot_torch(mask, 2).to(device)
        # the background mask is not fed into the model
        prediction = processor.step(frame_torch, mask_torch[1:])
      else:
        # propagate only
        prediction = processor.step(frame_torch)

      # argmax, convert to numpy
      prediction:np.ndarray = torch_prob_to_numpy_mask(prediction)

      # prediction_torch = prediction.transpose(2, 0, 1)
      print(prediction.shape)
      print(frame.shape)

      # TODO TEST BELOW CODE
      # prediction_torch = torch.from_numpy(prediction).float().to(device)/255
      # frame_torch = torch.from_numpy(frame).float().to(device)/255 
      # frame_norm = im_normalization(frame)
      
      # converting frame & prediction to torch for faster overlay
      # torch_frame = np.transpose(frame, (2, 0, 1))
      # torch_frame = torch.from_numpy(torch_frame)
      # torch_frame = torch_frame.float()/255.0
      # prediction_torch = np.expand_dims(prediction, axis=0)  # (480, 854) -> (1, 480, 854)
      # prediction_torch = torch.from_numpy(prediction_torch)
      # prediction_torch = prediction_torch.float()
      
      
      # visualization = overlay_davis_torch(torch_frame, prediction_torch)
      visualization = overlay_davis(frame, prediction)

      current_frame_index += 1
      mask_path = os.path.join(base_mask_path ,f"mask_{current_frame_index}.png")
      overlay_path = os.path.join(base_overlay_path ,f"overlay_{current_frame_index}.png")
      # save the mask and overlay      
      save_mask(dir=mask_path, mask=prediction)
      save_overlay(dir=overlay_path, overlay=visualization)
      
      torch.cuda.empty_cache()