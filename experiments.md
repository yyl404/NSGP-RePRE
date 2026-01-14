论文标题：Eigen-value Weighted Projection Regularization for Incremental Object Detection

# 需要补充的实验

## 1. 主实验

### 1.1 VOC（Faster RCNN）
1. 19+1
2. 15+5
3. 10+10

### 1.2 COCO（Faster RCNN）
1. 40+40
2. 70+10

## 2. 多阶段增量实验

### 2.1 VOC (Faster RCNN)
1. 5+5+5+5

## 3. 消融实验（以5+5+5+5和Faster RCNN为基准）

1. EWPR + Prototypes Replay
2. NSGL(Plastic) + Prototypes Replay
3. NSGL + Prototypes Replay
4. Prototypes Replay

### 3.1 有效性：1-3-4
||0-5|6-10|11-15|16-20|
|---|---|---|---|---|
|EWPR+RePRE|||||
|NSGL+RePRE|||||
|RePRE|||||

### 3.2 Stability：1-2-3

计算每一层的权重漂移大小（模型权重变化量在Core Feature Subpace的投影长度）平均值，记为$L_\text{drift}$

||EWPR|NSGL(Plastic)|NSGP|
|---|---|---|---|
|$L_\text{drift}$||||

### 3.3 Plasticity：1-2-3

比较新类别精度的上升速度（作一个折线图）
比较$L_\text{drift}$的变化趋势（作一个折线图）