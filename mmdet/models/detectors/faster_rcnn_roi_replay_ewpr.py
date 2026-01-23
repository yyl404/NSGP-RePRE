# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
import os
import os.path as osp

import numpy as np
import scipy
import scipy.ndimage
import torch
from torch import Tensor
from torchvision.ops import box_iou

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.logging import print_log
from . import TwoStageDetector


@MODELS.register_module()
class FasterRCNNRoIReplayEWPR(TwoStageDetector):
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
                 init_cfg: OptMultiConfig = None,
                 ewpr_loss_weight: float=10.0) -> None:
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
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        # EWPR (Eigen Weighted Projection Regularization) parameters
        # Each layer can have different (a, b) parameters for sigmoid reshaping function
        # Automatically computed based on adaptive_threshold
        self.ewpr_params = defaultdict(dict)  # {layer_name: {'a': float, 'b': float}}
        # EWPR loss weight for gradient computation (raw value will be used for logging)
        self.ewpr_loss_weight = ewpr_loss_weight
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
        
        # ========== MODIFICATION START: Compute EWPR (Eigen Weighted Projection Regularization) Loss ==========
        # Compute EWPR loss if teacher model and eigenvectors are available
        # Use offset=0.0 by default (same as adaptive_threshold default)
        if hasattr(self, "teacher_model") and len(self.eigens) > 0:
            ewpr_loss_raw = self.compute_ewpr_loss(offset=0.0)
            if ewpr_loss_raw is not None:
                # Store weighted loss for gradient computation (used in backward pass)
                losses['loss_ewpr'] = ewpr_loss_raw * self.ewpr_loss_weight
                # Store raw loss for logging (not used in gradient computation)
                # Note: Use non-'loss' prefix to avoid being included in gradient computation
                losses['ewpr_loss_raw'] = ewpr_loss_raw
        # ========== MODIFICATION END ==========
        
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

    def get_eigens(self, fea_in, distinguisher=None):
        """Compute eigenvalues and eigenvectors via SVD on covariance matrices.
        
        This function performs Singular Value Decomposition (SVD) on the covariance
        matrices stored in `fea_in` to extract the principal directions (eigenvectors)
        and their importance (eigenvalues) for NSGP gradient projection.
        
        Args:
            fea_in (dict[str, torch.Tensor]): Dictionary containing covariance matrices
                for each layer. The structure is:
                
                Structure:
                --------
                {
                    "layer_name.weight": torch.Tensor,  # 2D tensor, shape (C, C)
                    ...
                }
                
                Details:
                --------
                - Key (str): Full module path name with ".weight" suffix.
                  Examples:
                    - "backbone.conv1.weight"
                    - "neck.fpn_convs.0.weight"
                    - "rpn_head.rpn_conv.weight"
                
                - Value (torch.Tensor): 2D covariance matrix of input features,
                  shape (C, C), where C is the input feature dimension.
                  
                  For Linear layers:
                    C = input_features (e.g., 256, 512, 1024)
                    
                  For Conv2d layers:
                    C = kernel_size[0] * kernel_size[1] * in_channels
                    (e.g., for 3x3 conv with 64 in_channels: C = 3*3*64 = 576)
                
                Computation:
                --------
                Covariance matrices are computed in `update_cov()` as:
                    cov = X^T @ X
                where X has shape (N, C) after unfolding/reshaping input features.
                
                The covariance matrix is accumulated across all training samples:
                    fea_in[layer_name] = sum(cov_i for all batches i)
                
                Loading:
                --------
                Typically loaded from disk in `update_optim_transforms()`:
                    fea_in = torch.load("covariance.pth")
                
                Note: Keys in `fea_in` should already be filtered (e.g., ignoring
                classifier heads) before passing to this function.
                
            distinguisher (str, optional): Optional identifier for plotting/debugging.
                If provided, calls `plot_sval_figures()` to visualize singular values.
        
        Process:
        --------
        For each model parameter with a weight attribute:
            1. Construct key as "parameter_name.weight"
            2. Check if key exists in fea_in (skip if missing)
            3. Perform full SVD on covariance matrix: C = U Σ V^T
            4. Store eigenvalues (singular values) and eigenvectors (V)
            5. These are later used in `get_transforms()` to compute projection matrices
        
        Note:
        --------
        - `some=False` ensures full SVD decomposition (all singular values computed)
        - Eigenvalues are stored in descending order (largest first)
        - Missing layers in fea_in are skipped (they may have been filtered out)
        - Only parameters with weight attributes (Conv2d, Linear, etc.) are processed
        """
        # Iterate through all model parameters
        # Match parameter names with fea_in keys (which are "name.weight" format)
        for name, param in self.named_parameters():
            # Skip if parameter doesn't have weight attribute or not requires_grad
            if not param.requires_grad:
                continue
            
            # Construct the key as it appears in fea_in: "name.weight"
            # Only process weight parameters (bias and other params are skipped)
            if name.endswith('.weight'):
                # Check if this layer's covariance matrix exists in fea_in
                if name not in fea_in.keys():
                    continue
                
                # Perform SVD: fea_in[name] = U @ diag(eigen_value) @ eigen_vector^T
                # fea_in[name] shape: (C, C) - covariance matrix
                # eigen_value shape: (C,) - singular values in descending order
                # eigen_vector shape: (C, C) - right singular vectors (columns are eigenvectors)
                _, eigen_value, eigen_vector = torch.svd(fea_in[name], some=False)
                
                # Store eigenvalues and eigenvectors
                self.eigens[name] = {
                    'eigen_value': eigen_value,
                    'eigen_vector': eigen_vector
                }
        
        # Optional: plot singular value spectrum for visualization/debugging
        if distinguisher is not None:
            self.plot_sval_figures(self.eigens, distinguisher)
    
    def adaptive_threshold(self, svals: torch.Tensor, offset: float = 0):
        """Adaptively determine threshold to separate important vs null singular values.
        
        This function finds the "elbow point" in the singular value spectrum using
        second-order derivatives. The elbow represents the transition from high-variance
        (important) directions to low-variance (null space) directions.
        
        Args:
            svals (torch.Tensor): Singular values in descending order, shape (N,)
            offset (float): Adjustment to threshold index
                - Positive: move threshold right (preserve more basis, more conservative)
                - Negative: move threshold left (preserve fewer basis, more aggressive)
                - Range: [-1, 1]
        
        Returns:
            torch.Tensor: Boolean mask of shape (N,), True for singular values to preserve
        """
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        
        if len(points) >= 128:
            # Smooth the curve to find stable elbow (reduces noise in high dimensions)
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            # First derivative: measures rate of decay
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            # Second derivative: measures change in decay rate (curvature)
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            # Drop boundary points to avoid edge artifacts
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            # Find peak curvature (elbow point) and map back to original singular value
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            # For small dimensions, compute derivatives directly without smoothing
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        
        # Find index of threshold value (rightmost occurrence >= threshold)
        i_thres = np.arange(len(points))[points >= thres_val].max()
        
        # Apply offset adjustment: shift threshold by offset * i_thres positions
        if -1 <= offset <= 1:
            # Proportional adjustment: offset relative to current threshold
            i_thres = min(i_thres + int(offset * (i_thres)), len(points) - 1)
            i_thres = max(0, i_thres)
        else:
            # Absolute adjustment: shift by fixed number of positions
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        # Create boolean mask: True for indices >= i_thres (preserve these basis vectors)
        # Note: preserves from i_thres onwards, meaning larger singular values
        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx

    def compute_ewpr_loss(self, offset: float = 0.0) -> torch.Tensor:
        """Compute EWPR loss following the logic in ultralytics/ultralytics/engine/ewpr.py.

        For each layer with precomputed eigen decomposition:
        1) Flatten weight difference (W_current - W_teacher) to (out_dim, C)
        2) Project onto PCA components: proj = ΔW @ V      -> (out_dim, C)
        3) Scale by eigen values (adjusted): scaled = proj * adjusted_eigen_values
        4) Loss = mean( ||scaled||_2 ) over output dimension, multiplied by a factor (100)
        The final loss is averaged across all processed layers.
        """
        if not hasattr(self, "teacher_model") or len(self.eigens) == 0:
            return None

        total_loss = 0.0
        num_layers = 0

        teacher_state_dict = self.teacher_model.state_dict()

        for layer_name in self.eigens.keys():
            if not layer_name.endswith('.weight'):
                continue

            # Find corresponding module to keep gradients
            layer_module = None
            for name, module in self.named_modules():
                if name + '.weight' == layer_name:
                    layer_module = module
                    break
            if layer_module is None:
                continue
            if not isinstance(layer_module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue

            # Current weights with grads
            current_weight = layer_module.weight
            # Teacher weights (no grads)
            if layer_name not in teacher_state_dict:
                print_log(f"[EWPR] Skip layer '{layer_name}': not in teacher_state_dict", logger='current')
                continue
            teacher_weight = teacher_state_dict[layer_name]
            if current_weight.shape != teacher_weight.shape:
                print_log(f"[EWPR] Skip layer '{layer_name}': shape mismatch current {tuple(current_weight.shape)} vs teacher {tuple(teacher_weight.shape)}", logger='current')
                continue
            teacher_weight = teacher_weight.to(current_weight.device).to(current_weight.dtype)

            # Eigen components
            eigen_vectors = self.eigens[layer_name]['eigen_vector'].to(current_weight.device)
            eigen_values = self.eigens[layer_name]['eigen_value'].to(current_weight.device)

            # Adjust eigen values following ultralytics' logic:
            # 1) Find elbow point using adaptive_threshold
            # 2) Normalize elbow value to 1.0
            # 3) Values before/at elbow -> 1.0, after elbow -> scaled by same factor
            def _adjust_eigen_values(vals: torch.Tensor) -> torch.Tensor:
                # vals: 1D tensor (C,)
                # Use adaptive_threshold to find elbow point (returns boolean mask)
                # The first True position in the mask is the elbow point index
                preserve_mask = self.adaptive_threshold(vals, offset=0.0)
                # Find the first index where mask is True (elbow point)
                elbow_idx = torch.nonzero(preserve_mask, as_tuple=False)
                if elbow_idx.numel() > 0:
                    elbow_idx = elbow_idx[0, 0].item()
                else:
                    # If no True found, use the last index
                    elbow_idx = vals.shape[0] - 1
                
                elbow_idx = max(0, min(elbow_idx, vals.shape[0] - 1))
                elbow_val = vals[elbow_idx]
                eps = 1e-8
                scale_factor = 1.0 / (elbow_val + eps)
                adjusted = torch.ones_like(vals)
                if elbow_idx + 1 < vals.shape[0]:
                    adjusted[elbow_idx + 1:] = vals[elbow_idx + 1:] * scale_factor
                return adjusted

            adjusted_eigen_values = _adjust_eigen_values(eigen_values)

            # ΔW
            weight_diff = current_weight - teacher_weight

            # Flatten to (out_dim, C)
            if isinstance(layer_module, torch.nn.Linear):
                weight_diff_flat = weight_diff  # (out_features, in_features)
            else:  # Conv2d
                expected_eigen_dim = layer_module.kernel_size[0] * layer_module.kernel_size[1] * layer_module.in_channels
                if eigen_vectors.shape[0] != expected_eigen_dim:
                    print_log(f"[EWPR] Skip layer '{layer_name}': eigen dim {eigen_vectors.shape[0]} != expected {expected_eigen_dim}", logger='current')
                    continue
                weight_diff_flat = weight_diff.view(weight_diff.shape[0], -1)  # (out_channels, C)

            # Dimension check
            if weight_diff_flat.shape[1] != eigen_vectors.shape[0]:
                print_log(f"[EWPR] Skip layer '{layer_name}': projection dim mismatch {weight_diff_flat.shape[1]} vs {eigen_vectors.shape[0]}", logger='current')
                continue

            # 1) Project
            projections = torch.matmul(weight_diff_flat, eigen_vectors)  # (out_dim, C)
            # 2) Scale (broadcast eigen values)
            scaled = projections * adjusted_eigen_values.unsqueeze(0)  # (out_dim, C)
            # 3) Norm per output, mean, scale factor 100 (match ultralytics implementation)
            layer_loss = 100.0 * scaled.norm(dim=1).mean()

            total_loss += layer_loss
            num_layers += 1

        if num_layers == 0:
            return None

        return total_loss / num_layers
