import math
from collections import defaultdict
import scipy.ndimage
import torch
from torch.optim.optimizer import Optimizer
from mmengine.registry import OPTIMIZERS
from mmengine.logging import MessageHub, MMLogger, print_log
import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import os.path as osp

@OPTIMIZERS.register_module()
class SGDNSCL(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, nesterov=False, svd=False, thres=1.001, # thres exists but didn't work.
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
                        weight_decay=weight_decay, svd=svd,
                        thres=thres)
        super(SGDNSCL, self).__init__(params, defaults)
        
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.count = 0

    def __setstate__(self, state):
        super(SGDNSCL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('svd', False)
            group.setdefault('names', [])

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for n, p in zip(group['names'], group['params']):
                # if p.grad is None:
                #     continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                # print(svd, self.transforms)
                if svd and len(self.transforms) > 0 and n in self.transforms.keys():
                    if len(update.shape) == 4:
                        # the transpose of the manuscript
                        update_ = torch.mm(update.view(update.size(
                            0), -1), self.transforms[n])
                        update_ = update_.view_as(update)
                       
                    else:
                        update_ = torch.mm(update, self.transforms[n])
           
                    
                else:
                    update_ = update
                p.data.add_(update_)
        return loss

    def adaptive_threshold(self, svals: torch.Tensor, offset: float = 0):
        """Adaptively determine threshold to separate important vs null singular values.
        
        This function finds the "elbow point" in the singular value spectrum using
        second-order derivatives. The elbow represents the transition from high-variance
        (important) directions to low-variance (null space) directions.
        
        Mathematical Intuition:
        --------
        Singular values typically decay from large to small. The goal is to find where
        the decay rate changes significantly (the "elbow"). This is detected by:
        1. First derivative: diff_o1 = -gradient (negative because values decrease)
        2. Second derivative: diff_o2 = curvature (peaks at elbow point)
        
        Args:
            svals (torch.Tensor): Singular values in descending order, shape (N,)
            offset (float): Adjustment to threshold index
                - Positive: move threshold right (preserve more basis, more conservative)
                - Negative: move threshold left (preserve fewer basis, more aggressive)
                - Range: [-1, 1]
        
        Returns:
            torch.Tensor: Boolean mask of shape (N,), True for singular values to preserve
        
        Algorithm:
        --------
        For large dimensions (>= 128):
            1. Apply Gaussian smoothing to reduce noise
            2. Compute 1st order differences (gradient)
            3. Compute 2nd order differences (curvature)
            4. Drop 3% on each end to avoid boundary effects
            5. Find peak curvature = elbow point
        
        For small dimensions (< 128):
            Skip smoothing, directly compute derivatives
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
        
        # assert 0 <= offset < 1, offset
        # print(offset)
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


    def plot_sval_figures(self, svals_dict, distinguisher=None, offset = 0.0):
        plt.close()
        
        svals_dict_ary = {k: v['eigen_value'].cpu().numpy() for k, v in svals_dict.items()}
        
        
        fig, ax_list = plt.subplots(len(svals_dict_ary.keys())//4+1, 4)
        fig.set_figheight(60)
        fig.set_figwidth(15)
        for i, k in enumerate(svals_dict_ary.keys()):
            zero_idx = self.adaptive_threshold(svals_dict[k]['eigen_value'], offset=offset).cpu().numpy()
            points: np.ndarray = svals_dict_ary[k]
            i_thres = np.arange(len(points))[zero_idx].min()

            ax_list[i // 4, i % 4].plot(np.arange(i_thres + 1), points[:i_thres + 1], color='blue')
            ax_list[i // 4, i % 4].plot(np.arange(i_thres, len(points)), points[i_thres:], color='red')
            ax_list[i // 4, i % 4].set_title(k)
        if not osp.exists(save_dir := osp.join('./', 'figures')):
            os.mkdir(save_dir)
        fig.tight_layout()
        fig.savefig(osp.join(save_dir, save_name := f"svals_task{1}_{distinguisher}.png"))
        plt.close()

    def get_transforms(self, offset=0.0):
        """Compute null space projection transforms for gradient updates.
        
        This function implements the core of NSGP (Null Space Gradient Projection).
        It computes projection matrices that map gradients to the null space of
        previous tasks' feature covariance, preventing catastrophic forgetting.
        
        Mathematical Background:
        --------
        Given covariance matrix C from previous tasks, perform SVD: C = UΣV^T
        - Large singular values (top eigenvalues) correspond to directions with
          high variance in previous tasks' features - these are important to preserve
        - Small singular values correspond to null/low-variance directions - safe to update
        
        The transform matrix P = VV^T projects gradients into the span of important
        directions. The actual gradient update becomes: g' = g - Pg = g(I - P),
        which projects into the null space (orthogonal complement).
        
        Args:
            offset (float): Adjustment factor for adaptive thresholding.
                Positive offset: reserve more basis vectors (more conservative)
                Negative offset: reserve fewer basis vectors (more aggressive)
                Range: [-1, 1]
        
        Process:
        --------
        1. For each parameter with SVD enabled
        2. Use adaptive_threshold to find cutoff: separate important vs null directions
        3. Select eigenvectors corresponding to important singular values
        4. Compute projection matrix: P = VV^T (projects to important subspace)
        5. Store transform for use in step() during gradient updates
        """
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for n, p in zip(group['names'], group['params']):
                # if p.grad is None:
                #     continue
                if n not in self.eigens.keys():
                    print_log(f"missing keys: {n}")
                    continue
                # print(offset)
                # if 'neck' not in n:
                #     offset = 0
                # else:
                #     offset = -1
                # print(n, offset)
                
                # Adaptive threshold: finds the "elbow" in singular value spectrum
                # Returns boolean mask: True for important singular values to preserve
                ind = self.adaptive_threshold(self.eigens[n]['eigen_value'], offset=offset)
                
                # Log statistics about reserved basis
                # if (self.eigens[n]['eigen_value']==0).sum() != 0:
                print_log('{}: reserving basis {}/{}; cond: {}, radio:{}'.format(
                    n,
                    ind.sum(), self.eigens[n]['eigen_value'].shape[0],  # How many basis preserved
                    self.eigens[n]['eigen_value'][0] /
                    self.eigens[n]['eigen_value'][ind][0],  # Condition number
                    self.eigens[n]['eigen_value'][ind].sum(
                    ) / self.eigens[n]['eigen_value'].sum()  # Energy ratio preserved
                ))
                
                # Construct projection matrix P = VV^T
                # Step 1: Extract columns of V corresponding to important singular values
                # basis shape: (feature_dim, num_preserved_basis)
                basis = self.eigens[n]['eigen_vector'][:, ind]
                
                # inverse_transform = torch.pinverse(basis)
                # self.inverse_transform[n] = inverse_transform / inverse_transform.transpose(1,0)
                
                # Step 2: Compute projection matrix: P = VV^T
                # transform shape: (feature_dim, feature_dim)
                # This projects any vector onto the span of preserved basis vectors
                transform = torch.mm(basis, basis.transpose(1, 0))
                
                # Normalize backbone transforms to prevent numerical issues
                # (backbone layers have larger feature dimensions)
                if 'backbone' in n:
                    self.transforms[n] = transform / torch.norm(transform)
                else:
                    self.transforms[n] = transform
                
                # print(n, "transform - I", self.transforms[n] - torch.eye(transform.shape[0]).to(transform))
                
                # Detach to prevent gradients flowing through transform computation
                self.transforms[n].detach_()

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
                  These keys correspond to parameter names in `self.param_groups`.
                
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
                
            distinguisher (str, optional): Optional identifier for plotting/debugging.
                If provided, calls `plot_sval_figures()` to visualize singular values.
        
        Process:
        --------
        For each parameter group with SVD enabled:
            1. Check if layer name exists in fea_in (skip if missing)
            2. Perform full SVD on covariance matrix: C = U Σ V^T
            3. Store eigenvalues (singular values) and eigenvectors (V)
            4. These are later used in `get_transforms()` to compute projection matrices
        
        Note:
        --------
        - `some=False` ensures full SVD decomposition (all singular values computed)
        - Eigenvalues are stored in descending order (largest first)
        - Missing layers in fea_in are skipped with a warning
        """
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for n, p in zip(group['names'], group["params"]):
                # if p.grad is None:
                #     continue
                # Skip layers not present in covariance dictionary (may be filtered out)
                if n not in fea_in.keys():
                    continue
                # print(n)
                eigen = self.eigens[n]
                # print(fea_in[n].keys())
                # Perform SVD: fea_in[n] = U @ diag(eigen_value) @ eigen_vector^T
                # fea_in[n] shape: (C, C) - covariance matrix
                # eigen_value shape: (C,) - singular values in descending order
                # eigen_vector shape: (C, C) - right singular vectors (columns are eigenvectors)
                _, eigen_value, eigen_vector = torch.svd(fea_in[n], some=False)
                eigen['eigen_value'] = eigen_value
                # print(n, eigen_value)
                eigen['eigen_vector'] = eigen_vector
        
        # Optional: plot singular value spectrum for visualization/debugging
        if distinguisher != None:
            self.plot_sval_figures(self.eigens, distinguisher)
            

    def get_update(self, group, grad, p):
        nesterov = group["nesterov"]
        state = self.state[p]

        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['previous_grad'] = torch.zeros_like(p.data)

        exp_avg = state['previous_grad']
        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)
            
        if group['momentum'] != 0:
            if state['step'] > 1:
                exp_avg.mul_(group['momentum']).add_(1-group['dampening'], grad)
            else:
                exp_avg.add_(grad)
        # State initialization
            if group['nesterov']:
                grad.add_(group['momentum'], exp_avg)
            else:
                grad = exp_avg

        step_size = group['lr'] * grad
        update = - step_size
        return update



def get_angle(u, proj):
    '''
    u is the vector, v is the eigens
    '''
    # 计算 u 在基 v 上的投影
    # print(u.shape, v.shape)
    # proj = torch.mm(u, transform.to(u.device))
    # 计算投影的模长
    proj_norm = torch.norm(proj, dim=1)
    
    # 计算 u 的模长
    u_norm = torch.norm(u, dim=1)

    # 计算夹角的余弦值
    cos_theta = torch.diag(torch.mm(u, proj.t())) / (proj_norm * u_norm)
    # 计算夹角
    theta = torch.rad2deg(torch.acos(cos_theta))
    print(cos_theta)
    print(torch.acos(cos_theta))
    print(theta)

    return theta