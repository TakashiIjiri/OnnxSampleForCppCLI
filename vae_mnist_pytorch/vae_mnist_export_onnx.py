import sys
import os
import numpy as np
import warnings
import torch
from matplotlib import pyplot as plt
"""
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from gplvm_kanji import GaussianKernel, GPLVM
"""
import vae_mnist


model = vae_mnist.VAE()

model_path = "./model_fin.pth"
model.load_state_dict(torch.load(model_path))
decoder = model.decoder

model.eval()
decoder.eval() #多分不要

#export
dummy_input = torch.randn(1, 2)
torch.onnx.export(decoder, dummy_input, 'decoder.onnx', export_params=True, verbose=False)

print("Done")

#TODO TEST the onnx
#https://torch.classcat.com/2020/01/21/pytorch-1-4-tutorials-super-resolution-with-onnxruntime/
