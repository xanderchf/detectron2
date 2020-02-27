# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import build_backbone
from ..postprocessing import boundary_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["BddMultitaskModel", "BDD_HEADS_REGISTRY", "GeneralSemSegFPNHead", "build_bdd_sem_seg_head"]


BDD_HEADS_REGISTRY = Registry("BDD_HEADS")
"""
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class BddMultitaskModel(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.tasks = set(cfg.TASKS)

        if 'sem_seg' in self.tasks:
            self.sem_seg_head = build_bdd_sem_seg_head(cfg.MODEL.SEM_SEG_HEADS['SEM_SEG_HEAD'], self.backbone.output_shape())
        if 'lane' in self.tasks:
            self.lane_dir_head = build_bdd_sem_seg_head(cfg.MODEL.SEM_SEG_HEADS['LANE_DIRECTION_HEAD'], self.backbone.output_shape())
            self.lane_cont_head = build_bdd_sem_seg_head(cfg.MODEL.SEM_SEG_HEADS['LANE_CONTINUITY_HEAD'], self.backbone.output_shape())
            self.lane_cat_head = build_bdd_sem_seg_head(cfg.MODEL.SEM_SEG_HEADS['LANE_CATEGORY_HEAD'], self.backbone.output_shape())
        if 'drivable' in self.tasks:
            self.drivable_head = build_bdd_sem_seg_head(cfg.MODEL.SEM_SEG_HEADS['DRIVABLE_HEAD'], self.backbone.output_shape())
        if len(np.intersect1d(list(self.tasks), ['det', 'ins_seg', 'mot', 'mots'])) > 0:
            self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
            self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
            self.vis_period = cfg.VIS_PERIOD
            self.input_format = cfg.INPUT.FORMAT
        if 'ins_seg' in self.tasks:
            pass
        if 'mot' in self.tasks:
            pass
        if 'mots' in self.tasks:
            pass

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        results = dict()
        losses = dict()
        if "sem_seg" in self.tasks:
            targets = None
            if "sem_seg" in batched_inputs[0]:
                targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
                targets = ImageList.from_tensors(
                    targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
                ).tensor
            results["sem_seg"], loss = self.sem_seg_head(features, targets)
            losses.update(loss)
        if "lane" in self.tasks:
            lane_dir_targets = None
            lane_cont_targets = None
            lane_cat_targets = None
            if "lane" in batched_inputs[0]:
                lane_targets = [x["lane"].to(self.device) for x in batched_inputs]

                lane_dir_targets = ImageList.from_tensors(
                    [x[:, :, 0] for x in lane_targets], self.backbone.size_divisibility).tensor
                lane_cont_targets = ImageList.from_tensors(
                    [x[:, :, 1] for x in lane_targets], self.backbone.size_divisibility).tensor
                lane_cat_targets = ImageList.from_tensors(
                    [x[:, :, 2] for x in lane_targets], self.backbone.size_divisibility).tensor

            lane_dir_results, loss = self.lane_dir_head(features, lane_dir_targets)
            losses.update(loss)
            lane_cont_results, loss = self.lane_cont_head(features, lane_cont_targets)
            losses.update(loss)
            lane_cat_results, loss = self.lane_cat_head(features, lane_cat_targets)
            losses.update(loss)

            results["lane_dir"] = lane_dir_results
            results["lane_cont"] = lane_cont_results
            results["lane_cat"] = lane_cat_results
        if "drivable" in self.tasks:
            targets = None
            if "drivable" in batched_inputs[0]:
                targets = [x["drivable"].to(self.device) for x in batched_inputs]
                targets = ImageList.from_tensors(
                    targets, self.backbone.size_divisibility, self.drivable_head.ignore_value
                ).tensor
            results["drivable"], loss = self.drivable_head(features, targets)
            losses.update(loss)
        if "det" in self.tasks:
            pass
        if "ins_seg" in self.tasks:
            pass
        if "mot" in self.tasks:
            pass
        if "mots" in self.tasks:
            pass

        if self.training:
            return losses

        # process semantic segmentation
        processed_results = [dict() for _ in range(len(images))]
        for task, seg_results in results.items():
            if not task in ["sem_seg", "drivable", "lane_dir", "lane_cont", "lane_cat"]:
                continue
            for i, (result, input_per_image, image_size) in enumerate(zip(seg_results, batched_inputs, images.image_sizes)):
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                if task.startswith('lane'):
                    r = boundary_postprocess(result, image_size, height, width)
                else:
                    r = sem_seg_postprocess(result, image_size, height, width)
                processed_results[i][task] = r
        return processed_results


def build_bdd_sem_seg_head(model_cfg, input_shape):

    name = model_cfg.NAME
    return BDD_HEADS_REGISTRY.get(name)(model_cfg, input_shape)


@BDD_HEADS_REGISTRY.register()
class GeneralSemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, model_cfg, input_shape: Dict[str, ShapeSpec], name=''):
        super().__init__()

        # fmt: off
        self.in_features      = model_cfg.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = model_cfg.IGNORE_VALUE
        self.num_classes      = model_cfg.NUM_CLASSES
        conv_dims             = model_cfg.CONVS_DIM
        self.common_stride    = model_cfg.COMMON_STRIDE
        norm                  = model_cfg.NORM
        self.loss_weight      = model_cfg.LOSS_WEIGHT
        self.fg_weight        = model_cfg.FG_WEIGHT
        self.loss_id          = model_cfg.LOSS_ID
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, self.num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None, masks=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            weight = None if self.fg_weight == 1 else torch.Tensor([1] + [self.fg_weight for _ in range(self.num_classes - 1)]).to(x.device)
            if masks:
                # mask out background if needed
                for target, mask in zip(targets, masks):
                    target[1 - mask] = 255
            losses["loss_{}".format(self.loss_id)] = (
                F.cross_entropy(x, targets, weight=weight, reduction="mean", ignore_index=self.ignore_value)
                * self.loss_weight
            )
            return [], losses
        else:
            return x, {}
