import copy

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.core.post_processing import nms_bev
from mmdet3d.models import builder
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import build_bbox_coder, multi_apply
from ..builder import HEADS, build_loss
from .centerpoint_head import CenterHead

@HEADS.register_module()
class FusionHead(CenterHead):
    def __init__(self,
                in_channels=[128],
                tasks=None,
                train_cfg=None,
                test_cfg=None,
                bbox_coder=None,
                common_heads=dict(),
                loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                loss_bbox=dict(
                    type='L1Loss', reduction='none', loss_weight=0.25),
                separate_head=dict(
                    type='SeparateHead', init_bias=-2.19,   final_kernel=3),
                share_conv_channel=64,
                num_heatmap_convs=2,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                bias='auto',
                norm_bbox=True,
                init_cfg=None):
        super().__init__(in_channels, tasks, train_cfg, test_cfg, bbox_coder, common_heads, loss_cls, loss_bbox, separate_head, share_conv_channel, num_heatmap_convs, conv_cfg, norm_cfg, bias, norm_bbox, init_cfg)
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # only loss for bounding box regression and classification
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        pass