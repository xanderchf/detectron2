import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

from ..roi_heads.box_head import FastRCNNConvFCHead

PROP_TRACK_HEAD_REGISTRY = Registry("PROP_TRACK_HEAD")
PROP_TRACK_HEAD_REGISTRY.__doc__ = """
Registry for prop heads.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def prop_bbox_targets(pos_bboxes_list,
                      pos_gt_pids_list,
                      pos_ref_bboxes_list,
                      cfg,
                      reg_classes=1,
                      target_means=[.0, .0, .0, .0],
                      target_stds=[1.0, 1.0, 1.0, 1.0],
                      concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        prop_bbox_targets_single,
        pos_bboxes_list,
        pos_gt_pids_list,
        pos_ref_bboxes_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def prop_bbox_targets_single(pos_bboxes,
                             pos_gt_pids,
                             pos_ref_bboxes,
                             cfg,
                             reg_classes=1,
                             target_means=[.0, .0, .0, .0],
                             target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_samples = pos_bboxes.size(0)
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    pos_inds = torch.nonzero(pos_gt_pids >= 0).view(-1)
    neg_inds = torch.nonzero(pos_gt_pids < 0).view(-1)
    num_pos = pos_inds.size(0)
    num_neg = neg_inds.size(0)
    assert num_pos + num_neg == num_samples
    if num_pos > 0:
        labels[pos_inds] = 1
        label_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = bbox2delta(pos_bboxes[pos_inds],
                                            pos_ref_bboxes[pos_inds],
                                            target_means, target_stds)
        bbox_weights[pos_inds] = 1.0
    if num_neg > 0:
        label_weights[neg_inds] = 1.0
    return labels, label_weights, bbox_targets, bbox_weights


@PROP_TRACK_HEAD_REGISTRY.register()
class PropTrackHead(FastRCNNConvFCHead):
    """Propogate bounding boxes from previous frame to current frame.
    Implementation is based on correlation and ConvFCBBoxHead.

    This head has two outputs: classification confidence and regression delta.
    The classification confidence means whether there is an object or not.
    The regression delta means the distance correspond to objects in the
        previous frame.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super(PropTrackHead, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.reg_class_agnostic = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.target_means = cfg.MODEL.PIXEL_MEAN
        self.target_std = cfg.MODEL.PIXEL_STD
        self.use_sigmoid_cls = True

    def get_target(self, sampling_results, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_gt_pids = [res.pos_gt_pids for res in sampling_results]
        pos_ref_bboxes = [res.pos_ref_bboxes for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        assert reg_classes == 1, "only class_agnostic considered now."
        prop_targets = prop_bbox_targets(
            pos_proposals,
            pos_gt_pids,
            pos_ref_bboxes,
            cfg=rcnn_train_cfg,
            reg_classes=reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)

        return prop_targets

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_prop_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['prop_acc'] = accuracy(
                cls_score, labels, sigmoid=self.use_sigmoid_cls)
        if bbox_pred is not None:
            if not self.reg_class_agnostic:
                bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
            losses['loss_prop_reg'] = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                avg_factor=bbox_targets.size(0) * 4,
                reduction_override=reduction_override)
        return losses

    def get_track_bboxes(self,
                         rois,
                         cls_score,
                         bbox_pred,
                         img_shape,
                         scale_factor,
                         rescale=False,
                         cfg=None):
        if self.use_sigmoid_cls:
            scores = torch.sigmoid(
                cls_score) if cls_score is not None else None
        else:
            scores = F.softmax(
                cls_score,
                dim=1)[:, 1][:, None] if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            bboxes = torch.cat([bboxes, scores], dim=1)
            return bboxes


def build_prop_track_head(cfg, input_shape):
    """
    Build a prop head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.PROP_TRACK_HEAD.NAME
    return PROP_TRACK_HEAD_REGISTRY.get(name)(cfg, input_shape)
