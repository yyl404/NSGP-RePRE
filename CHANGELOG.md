# 项目修改日志 (Changelog)

本文档记录了 NSGP-RePRE 项目在开发过程中所做的所有代码修改以及解决的问题。

## 修改概览

本项目的修改主要集中在以下几个方面：
1. 修复分布式/非分布式训练兼容性问题
2. 修复配置文件参数读取问题
3. 添加缺失的方法实现
4. 修复特征形状不匹配问题
5. 修复硬编码问题，提高代码通用性

---

## 详细修改记录

### 1. 修复 `AttributeError: 'FasterRCNNRoIReplay' object has no attribute 'module'` 错误

**文件**: `mmdet/engine/runner/nsrunner_roi_replay.py`

**问题描述**: 
在非分布式训练模式下，模型没有被 `MMDistributedDataParallel` 或 `DistributedDataParallel` 包装，因此没有 `.module` 属性。代码中多处直接访问 `self.model.module` 导致 `AttributeError`。

**解决方案**:
- 在所有访问 `self.model.module` 的地方添加 `is_model_wrapper()` 判断
- 如果模型被包装，使用 `self.model.module`；否则直接使用 `self.model`

**修改位置**:
1. **第288行** - 获取模型名称时：
   ```python
   # 修改前
   if hasattr(self.model, 'module'):
       self._model_name = self.model.module.__class__.__name__
   
   # 修改后
   if is_model_wrapper(self.model):
       self._model_name = self.model.module.__class__.__name__
   ```

2. **第774-781行** - `calculate_save_importance` 方法中：
   ```python
   # 修改前
   data = self.model.module.data_preprocessor(data_batch, True)
   parsed_losses, log_vars = self.model.module.parse_losses(losses)
   
   # 修改后
   if is_model_wrapper(self.model):
       model = self.model.module
   else:
       model = self.model
   data = model.data_preprocessor(data_batch, True)
   parsed_losses, log_vars = model.parse_losses(losses)
   ```

**影响**: 
- ✅ 解决了非分布式训练时的 `AttributeError` 错误
- ✅ 代码现在同时兼容分布式和非分布式训练模式
- ✅ 提高了代码的健壮性

---

### 2. 修复配置文件参数 `ckpt_keywords` 无法读取的问题

**文件**: `mmdet/engine/runner/nsrunner_roi_replay.py`

**问题描述**: 
`from_cfg` 方法在从配置文件构建 runner 时，没有读取 `ckpt_keywords` 参数，导致即使配置文件中指定了该参数也无法生效。

**解决方案**:
在 `from_cfg` 方法中添加 `ckpt_keywords` 参数的读取。

**修改位置**:
- **第366行** - 在 `from_cfg` 方法中：
  ```python
  # 修改前
  previous_dir=cfg.get('previous_dir'),
  cfg=cfg,
  
  # 修改后
  previous_dir=cfg.get('previous_dir'),
  ckpt_keywords=cfg.get('ckpt_keywords'),
  cfg=cfg,
  ```

**影响**:
- ✅ 用户现在可以在配置文件中使用 `ckpt_keywords` 参数
- ✅ 支持通过关键字自动匹配 checkpoint 文件
- ✅ 提高了配置的灵活性

**使用示例**:
```python
# 在配置文件中
ckpt_keywords = "epoch_12"  # 匹配包含 "epoch_12" 的checkpoint文件
# 或
ckpt_keywords = "best"      # 匹配包含 "best" 的checkpoint文件
```

---

### 3. 添加缺失的 `get_mid_features` 方法并修复特征形状问题

**文件**: `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_task.py`

**问题描述**: 
1. `standard_roi_replay_head.py` 中调用了 `self.bbox_head.get_mid_features()` 方法，但该方法在 `Shared2FCBBoxHeadTask` 及其父类 `ConvFCBBoxHeadTask` 中不存在，导致 `AttributeError: 'Shared2FCBBoxHeadTask' object has no attribute 'get_mid_features'`。
2. 后续代码（`standard_roi_replay_head.py` 第417行）期望特征形状为 `(N, 7*7*256) = (N, 12544)`，用于原型计算和相似度计算。

**解决方案**:
在 `ConvFCBBoxHeadTask` 类中添加 `get_mid_features` 方法，该方法返回 flatten 后但未经过共享FC层的特征，形状为 `(N, 12544)`。

**修改位置**:
- **第290-323行** - 添加新方法：
  ```python
  def get_mid_features(self, x: Tensor) -> Tensor:
      """Extract intermediate features before shared FC layers.
      
      This method extracts features after flattening but before passing 
      through shared FC layers. For Shared2FCBBoxHeadTask, this corresponds 
      to features after flattening the RoI features (7*7*256 = 12544).
      
      Args:
          x (Tensor): Input features from bbox_roi_extractor and 
              optional shared_head, shape (N, C, H, W).
      
      Returns:
          Tensor: Features after flattening but before shared FC layers.
              For Shared2FCBBoxHeadTask, this is flattened RoI features
              with shape (N, 7*7*256) = (N, 12544).
      """
      # shared part - process conv layers if any
      if self.num_shared_convs > 0:
          for conv in self.shared_convs:
              x = conv(x)
      
      # For shared FC layers, return features after flattening but before FC
      if self.num_shared_fcs > 0:
          if self.with_avg_pool:
              x = self.avg_pool(x)
          
          # Flatten and return (don't pass through FC layers)
          x = x.flatten(1)
          return x
      else:
          # If no shared FCs, just flatten if needed
          if x.dim() > 2:
              x = x.flatten(1)
          return x
  ```

**方法功能**:
- 提取经过共享卷积层（如果有）处理后的特征
- 对于共享FC层，返回 flatten 后但**未经过FC层**的特征
- 对于 `Shared2FCBBoxHeadTask`（`roi_feat_size=7`, `in_channels=256`），返回形状为 `(N, 12544)` 的特征

**特征处理流程**:
1. `bbox_roi_extractor` 提取RoI特征：形状 `(N, 256, 7, 7)`
2. 经过 `shared_head`（如果有）
3. `get_mid_features` 处理：
   - 平均池化（如果启用）
   - Flatten：形状变为 `(N, 12544)` = `(N, 7*7*256)`
   - **不经过FC层，直接返回**
4. 保存到 `rois_etc.pth`，供后续原型计算使用

**影响**:
- ✅ 解决了 `AttributeError: 'Shared2FCBBoxHeadTask' object has no attribute 'get_mid_features'` 错误
- ✅ 解决了 `RuntimeError: shape '[-1, 12544]' is invalid for input of size 242688` 错误
- ✅ 特征形状与后续处理代码期望的形状一致（`standard_roi_replay_head.py` 第417行的 `reshape(-1, 7*7*256)` 可以正常工作）
- ✅ 为 RePRE 方法的原型重放功能提供了正确的特征提取

---

### 4. 修复背景类别ID硬编码问题

**文件**: `mmdet/models/roi_heads/standard_roi_replay_head.py`

**问题描述**: 
代码中硬编码了背景类别ID的判断逻辑，使用了未定义的变量 `VOC`，导致 `NameError`。代码假设VOC数据集背景类别ID为20，COCO为80，但应该根据实际的 `num_classes` 动态判断。

**解决方案**:
使用 `self.bbox_head.num_classes` 动态获取背景类别ID（背景类别ID等于前景类别数）。

**修改位置**:
- **第163-166行**：
  ```python
  # 修改前（会报 NameError）
  mask = cls_target != 20 if VOC else 80
  
  # 修改后
  # Filter out background class: background class ID equals num_classes
  # (VOC: num_classes=20, background=20; COCO: num_classes=80, background=80)
  bg_class_id = self.bbox_head.num_classes
  mask = cls_target != bg_class_id
  ```

**影响**:
- ✅ 解决了 `NameError: name 'VOC' is not defined` 错误
- ✅ 代码现在自动适配不同的数据集（VOC、COCO等）
- ✅ 提高了代码的通用性和可维护性

**原理说明**:
在目标检测任务中，类别索引从0开始：
- VOC数据集：20个前景类（0-19），背景类ID = 20
- COCO数据集：80个前景类（0-79），背景类ID = 80
- 通用规则：背景类ID = `num_classes`

---

## 修改总结

| 序号 | 问题类型 | 文件 | 行号 | 状态 |
|------|---------|------|------|------|
| 1 | AttributeError (module属性) | `nsrunner_roi_replay.py` | 288, 774-781 | ✅ 已修复 |
| 2 | 配置参数无法读取 | `nsrunner_roi_replay.py` | 366 | ✅ 已修复 |
| 3 | AttributeError + RuntimeError (缺失方法+形状不匹配) | `convfc_bbox_head_task.py` | 290-323 | ✅ 已修复 |
| 4 | NameError (未定义变量) | `standard_roi_replay_head.py` | 163-166 | ✅ 已修复 |

---

## 测试建议

完成以上修改后，建议进行以下测试：

1. **非分布式训练测试**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/xxx.py
   ```
   验证不会出现 `AttributeError: 'FasterRCNNRoIReplay' object has no attribute 'module'` 错误

2. **配置文件参数测试**：
   在配置文件中添加 `ckpt_keywords = "epoch_12"`，验证能正确匹配checkpoint文件

3. **特征提取测试**：
   验证 `get_mid_features` 返回的特征形状为 `(N, 12544)`，且能正确用于原型计算

4. **不同数据集测试**：
   分别在VOC和COCO数据集上测试，验证背景类别ID能正确识别

---

## 注意事项

1. **特征形状**：
   - `get_mid_features` 返回的特征形状取决于 `roi_feat_size` 和 `in_channels`
   - 对于标准配置（`roi_feat_size=7`, `in_channels=256`），特征形状为 `(N, 12544)`

2. **分布式训练**：
   - 修改后的代码同时支持分布式和非分布式训练
   - 使用 `is_model_wrapper()` 判断模型是否被包装

3. **配置文件**：
   - `ckpt_keywords` 是可选参数
   - 如果未指定，`cfg.get('ckpt_keywords')` 返回 `None`，行为与之前一致

---

## 相关文件

- `mmdet/engine/runner/nsrunner_roi_replay.py` - Runner实现
- `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_task.py` - BBox Head实现
- `mmdet/models/roi_heads/standard_roi_replay_head.py` - RoI Head实现
- `cl_faster_rcnn_cfgs/incremental_task/cl_faster_rcnn_nsgp_repre_*.py` - 配置文件示例

---

**最后更新**: 2026-01-04
