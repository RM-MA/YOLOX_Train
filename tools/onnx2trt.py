#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch

model_path = '/home/yiyu/py_code/YOLOX/armor-nano-poly.onnx'
trt_path = 'test.engine'
x = torch.ones(1, 3, 416, 416)
TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH= [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
builder = trt.Builder(TRT_LOGGER)
network =  builder.create_network(*EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)
    # builder.max_workspace_size = 1
builder.max_batch_size = 1
print(network)
with open(model_path, 'rb') as model:
    if not parser.parse(model.read()):
        print(parser.num_errors)
last_layer = network.get_layer(network.num_layers - 1)
network.mark_output(last_layer.get_output(0))

network.get_input(0).shape = [1, 3, 416, 416]
engine = builder.build_engine(network)
with open(trt_path, 'wb') as f:
    f.write(engine.serialize())
logger.info("Converted TensorRT model engine file is saved for C++ inference.")


@logger.catch
@torch.no_grad()
def main():
    model_path = '/home/yiyu/py_code/YOLOX/armor-nano-poly.onnx'
    trt_path = 'test.engine'
    x = torch.ones(1, 3, 416, 416)

    TRT_LOGGER = trt.Logger()

    EXPLICIT_BATCH = [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(*EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # builder.max_workspace_size = 1
    builder.max_batch_size = 1
    print(network)
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(parser.num_errors)
    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))

    network.get_input(0).shape = [1, 3, 416, 416]
    engine = builder.build_cuda_engine(network)

    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


# if __name__ == "__main__":
#     main()
