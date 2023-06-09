#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn
import torch

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.test_size = (416,416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # --------------- transform config ----------------- #
        # # Mosaic
        # self.mosaic_prob = 0.0
        # self.mosaic_scale = (0.3, 2)
        # # Mixup
        # self.enable_mixup = False
        # self.mixup_prob = 0.0
        # self.mixup_scale = (0.5, 1.5)
        # # HSV
        # self.hsv_prob = 0.0
        # # Gaussian Blur
        # self.noise_prob = 0.5
        # # Flip
        # self.flip_prob = 0.0
        # # Affine
        # self.degrees = 20.0
        # self.translate = 1.0
        # self.shear = 1.0
        # self.perspective = 1.0

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            in_channels_head = [64, 128, 256]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True, act=self.act)
            head = YOLOXHead(self.num_apexes, self.num_classes, self.num_colors, self.width, in_channels=in_channels_head, depthwise=True, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
