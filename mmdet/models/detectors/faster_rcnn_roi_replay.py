# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch import Tensor
from torchvision.ops import box_iou

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from . import TwoStageDetector


@MODELS.register_module()
class FasterRCNNRoIReplay(TwoStageDetector):
    """ A modified version of `Faster R-CNN` form original version in mmdet-3.3.0
    This version is extended with pseudo-labels generation for incremental learning.
    """

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        # ========== MODIFICATION START: Add threshold parameters for pseudo-labels generation ==========
        self.rpn_thresh = 0.5
        self.roi_thresh = 0.7
        # ========== MODIFICATION END ==========


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             use_teacher_student: bool=True) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            use_teacher_student (bool): Whether to use teacher-student
                pseudo-labeling for incremental learning. If True, the teacher
                model (from previous task) generates pseudo-labels to augment
                training data for both RPN and RoI head stages. Defaults to True.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        
        # ========== MODIFICATION START: Generate pseudo-labels from teacher model for incremental learning ==========
        rpn_data_samples = None
        if hasattr(self, "teacher_model") and use_teacher_student:
            with torch.no_grad():
                self.teacher_model.eval()
                
                # Generate predictions from teacher model
                # Note: teacher_model's task_id is set to (current_task_id - 1) during initialization
                teacher_predictions = self.teacher_model.predict(
                    batch_inputs, copy.deepcopy(batch_data_samples), rescale=False)
                
                # Initialize data samples for RPN and RoI head with pseudo-labels
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                
                # Filter and augment pseudo-labels based on IoU and confidence thresholds
                for teacher_result, gt_data_sample, rpn_data_sample in zip(
                        teacher_predictions, batch_data_samples, rpn_data_samples):
                    
                    for pseudo_bbox in teacher_result.pred_instances:
                        # Calculate maximum IoU with all ground truth boxes
                        # box_iou returns a matrix (1, num_gt), take the maximum
                        if len(gt_data_sample.gt_instances) > 0:
                            iou_matrix = box_iou(pseudo_bbox.bboxes, gt_data_sample.gt_instances.bboxes)
                            max_iou = iou_matrix.max().item()
                        else:
                            max_iou = 0.0
                        
                        # Skip pseudo-labels with high IoU (>0.7) to avoid redundant annotations
                        if max_iou > 0.7:
                            continue
                        
                        # Extract confidence score and remove it from bbox for concatenation
                        confidence_score = pseudo_bbox['scores']
                        pseudo_bbox.__delattr__('scores')
                        
                        # Add pseudo-label to RPN training data if confidence > rpn_thresh
                        if confidence_score > self.rpn_thresh:
                            rpn_data_sample.gt_instances = rpn_data_sample.gt_instances.cat(
                                [rpn_data_sample.gt_instances, pseudo_bbox])
                        
                        # Add pseudo-label to RoI head training data if confidence > roi_thresh
                        if confidence_score > self.roi_thresh:
                            gt_data_sample.gt_instances = gt_data_sample.gt_instances.cat(
                                [gt_data_sample.gt_instances, pseudo_bbox])
        # ========== MODIFICATION END ==========
        
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = rpn_data_samples if rpn_data_samples else copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        
        return losses


    # ========== MODIFICATION START: Add get_bbox_stuff method for RePRE ==========
    def get_bbox_stuff(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            _, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        
        # Extract RoI features and associated targets for RePRE (Regional Prototype Replay)
        # Returns: (bbox_feats, cls_target, cls_weight, bbox_target, bbox_weight, rois)
        roi_replay_data = self.roi_head.get_bbox_stuff(x, rpn_results_list, 
                                                       batch_data_samples)

        return roi_replay_data
    # ========== MODIFICATION END ==========
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        # ========== MODIFICATION START: New modes for NSGP-RePRE ==========
        elif mode == 'nullspace':
            return self.loss(inputs, data_samples, use_teacher_student=False) # For NSGP.
        elif mode == 'roi_replay':
            return self.get_bbox_stuff(inputs, data_samples)  # For RePRE Regional feature computation.
        # ========== MODIFICATION END ==========
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
            
    # ========== MODIFICATION START: Override predict method ==========
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)
        _, rpn_results_list = self.rpn_head.loss_and_predict(
            x, batch_data_samples, proposal_cfg=proposal_cfg)

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    # ========== MODIFICATION END ==========