# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

import torchvision.ops as ops
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet.core import multi_apply
from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector
import numpy as np
from .mvx_two_stage import MVXTwoStageDetector

@DETECTORS.register_module()
class RCFusion(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 img_bbox_head=None,
                 fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RCFusion, self).__init__(init_cfg=init_cfg)

        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None: 
            self.img_roi_head = builder.build_head(img_roi_head)
        if img_bbox_head is not None: # for cenetrnet head
            img_train_cfg = train_cfg.get('img') if train_cfg else None
            img_bbox_head.update(train_cfg=img_train_cfg)
            img_test_cfg = test_cfg.get('img') if test_cfg else None
            img_bbox_head.update(test_cfg=img_test_cfg)
            self.img_bbox_head = builder.build_head(img_bbox_head)
            
            

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_roi_head.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def crop_img_feats(self, list_of_boxes, img_feats, is_train=False):
        """
        crop image feats based on 2D detection boxes
        Args:
            list_of_boxes list(tuple[Tensor, Tensor]): The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
            img_feats tuple(Tensor): The image featues in shape [B, C, H, W]
        """
        bs = img_feats[0].shape[0]
        feat_map = img_feats[0]
        boxes = []
        for i in list_of_boxes:
            temp_box = [i[0][:,:-1]]
            # drop boxes or add boxes to maintain same size during training
            train_sample = 100
            if is_train:
                if temp_box[0].shape[0] < train_sample:
                    # pad existing sample
                    raise NotImplementedError('now default setting would return 100 during training.')
                elif temp_box[0].shape[0] > train_sample:
                    # delete extra sample
                    raise NotImplementedError('now default setting would return 100 during training.')
            else:
                assert len(list_of_boxes)==1, "only batch_size == 1 is allowed during test time!"
            boxes += temp_box
        pass
    
        roi_size = (7, 7) # put this to config later
        output_feats = ops.roi_align(feat_map, boxes, output_size=roi_size, spatial_scale=1.0)
        _, c, h, w = output_feats.shape
        output_feats = output_feats.view([bs, -1, c, h, w])
        return output_feats
    
    @staticmethod
    def xywh_to_tlbr(boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        tl_x = x - w/2
        tl_y = y - h/2
        br_x = x + w/2
        br_y = y + h/2
        return torch.stack([tl_x, tl_y, br_x, br_y], dim=1)
    
    def convert_to_bev_pixel(self, boxes):
        """
        Args:
            boxes (Tensor): [N, 4] in [tl_x, tl_y, br_x, br_y]
        """
        point_cloud_range = self.pts_bbox_head.bbox_coder.pc_range
        voxel_size = self.pts_bbox_head.bbox_coder.voxel_size
        out_size_factor = self.pts_bbox_head.bbox_coder.out_size_factor
        x_offset = point_cloud_range[0]
        y_offset = point_cloud_range[1]
        x_voxel_factor = voxel_size[0] * out_size_factor
        y_voxel_factor = voxel_size[1] * out_size_factor
        tl_x, tl_y, br_x, br_y = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        tl_x = (tl_x - x_offset) / x_voxel_factor
        br_x = (br_x - x_offset) / x_voxel_factor
        tl_y = (tl_y - y_offset) / y_voxel_factor
        br_y = (br_y - y_offset) / y_voxel_factor
        
        return torch.stack([tl_x, tl_y, br_x, br_y], dim=1)
    
    def crop_pts_feats(self, list_of_boxes, pts_feats, img_metas, is_train=False):
        """
         crop image feats based on 2D detection boxes
        Args:
            list_of_boxes dict(List(Tensor)): len(List) is the batch size, inside the tensor shows number of boxes detected, 
                the feature will be cropped in the BEV space for the sake of similicity.
            img_feats tuple(Tensor): The image featues in shape [B, C, H, W]
        """
        boxes_bev = list_of_boxes['bbox_bev']
        boxes_bev_cat = torch.cat([x for x in boxes_bev], dim=0)
        boxes_bev_cat_roi = self.xywh_to_tlbr(boxes_bev_cat) # still in 3D space coordinate
        batch_size = pts_feats[0].shape[0]
        feats = pts_feats[0]
        roi_size = (7, 7) # put this to config later
        output_feats = ops.roi_align(feats, boxes_bev_cat_roi, output_size=roi_size, spatial_scale=1.0)
        return output_feats
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    def extract_bboxes_2d(self,
                          img,
                          img_metas,
                          img_feats=None,
                          train=True,
                          bboxes_2d=None,
                          **kwargs):
        """Extract bounding boxes from 2d detector.

        Args:
            img (torch.Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): Image meta info.
            img_feats (torch.Tensor): extracted image feats
            train (bool): train-time or not.
            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,
                not supported yet.

        Return:
            list[torch.Tensor]: a list of processed 2d bounding boxes.
        """
        if bboxes_2d is None:
            if img_feats is None:
                x = self.extract_img_feat(img)
            else:
                x = img_feats
                
            # =================== for centernet
            # get 2d box
            with_nms = not train
            center_heatmap_pred, wh_preds, offset_preds = self.img_bbox_head(x)
            list_of_boxes = self.img_bbox_head.get_bboxes(center_heatmap_pred, wh_preds, offset_preds, img_metas, rescale=False, with_nms=with_nms) # bboxes in feats scale
            return list_of_boxes
        
    
    @staticmethod
    def fix_num_boxes(boxes, fix_num, idx=None):
        """
        Args:
            boxes (Tensor): (N, K) tensor containing boxes
            fix_num (int): number of output boxes
            idx (Tensor): whether to reuse index tensor for box dropping
        """
        box_num = boxes.shape[0]
        if box_num >= fix_num:
            if idx is not None:
                return boxes[idx, :], idx
            else:
                indices = torch.randperm(box_num)[:fix_num]
                return boxes[indices, :], indices
        else:
            # pad the first box multiple times
            repeat_times = fix_num - box_num
            final_boxes = torch.cat([boxes, boxes[0, :].repeat(repeat_times, 1)], dim=0)
            return final_boxes, None
    
    @torch.no_grad()
    def extract_bboxes_3d(self,
                          pts,
                          img_metas,
                          pts_feats=None,
                          train=True,
                          bboxes_3d=None,
                          rotated_roi=False,
                          fix_roi_num=200,
                          **kwargs):
        """Extract bounding boxes from 2d detector.

        Args:
            pts (torch.Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): Image meta info.
            pts_feats (torch.Tensor): extracted image feats
            train (bool): train-time or not.
            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,
                not supported yet.
            fix_roi_num (int): output a fix number of roi. Pad zero vector to the original one if less, random drop if more

        Return:
            dict(): a dictionary with bbox in 3D and BEV
        """
        if bboxes_3d is None:
            if pts_feats is None:
                x = self.extract_pts_feat(pts)
            else:
                x = pts_feats
                
            # =================== for centernet
            # get 2d box
            with_nms = not train
            result_dict = self.pts_bbox_head(x)
            pred_3d_boxes = self.pts_bbox_head.get_bboxes(result_dict, img_metas) # bboxes in feats scale
            # change the 3D box to 2D
            #     proposal_list = [
            #     dict(
            #         boxes_3d=bboxes,
            #         scores_3d=scores,
            #         cls_preds=preds_cls)
            #     for bboxes, scores, preds_cls in pred_3d_boxes
            # ]
            bbox_results = [
            bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in pred_3d_boxes
            ]
            # gather bev coordinate
            boxes_bev_list = []
            boxes_3d_list = []
            for bbox_dict in bbox_results:
                box_inst = bbox_dict['boxes_3d']
                box_loc_dim_rot = box_inst.tensor[:,:7] # x, y, z, x_size, y_size, z_size, yaw
                score_temp = bbox_dict['scores_3d'].reshape([-1, 1])
                label_temp = bbox_dict['labels_3d'].reshape([-1, 1])
                boxes3d_temp = torch.cat((box_loc_dim_rot, score_temp, label_temp), dim=1)
                nearest_bev, indices = self.fix_num_boxes(box_inst.nearest_bev, fix_roi_num)
                if indices is not None:
                    boxes3d_temp = self.fix_num_boxes(boxes3d_temp, fix_roi_num, idx=indices)
                else:
                    boxes3d_temp = self.fix_num_boxes(boxes3d_temp, fix_roi_num)
                boxes_bev_list += [nearest_bev]
                boxes_3d_list += [boxes3d_temp]
                
            if rotated_roi:
                raise NotImplementedError('roi align for rotated box in BEV perspective will be supported in future version.')
            else:
                
                pass
            
            bbox_ret = {
                'bbox_bev': boxes_bev_list,
                'bbox_3d': boxes_3d_list
            }
            return bbox_ret
        
        
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        # pass img feats to forward_img_train
        img_boxes = self.extract_bboxes_2d(img=img, img_metas=img_metas, img_feats=img_feats, train=True)
        
        # crop the image feature map based on predicted 2D RoI
        img_roi_feats = self.crop_img_feats(img_boxes, img_feats, is_train=True)
        
        # crop the pts feature map based on 3D RoI (we don't need this if we use point-based backbone)
        
        pts_boxes = self.extract_bboxes_3d(pts=points, img_metas=img_metas, pts_feats=pts_feats)
        pts_roi_feats = self.crop_pts_feats(pts_boxes, pts_feats, is_train=True)
        
        # pass img_feats and predicted RoI and pts_feats to fusion layer
        
        # pass fused feats to forward_fusion_train
        
        
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[torch.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[torch.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            proposals (list[torch.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                                 self.test_cfg.img_rpn)
        else:
            proposal_list = proposals

        return self.img_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN test function."""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)
