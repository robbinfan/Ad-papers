# ESMM 图解笔记

> **论文**: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (Alibaba 2018, SIGIR)

---

## 一、要解决什么问题？

传统 CVR 建模有两个痛点：

```
┌─────────────────────────────────────────────────┐
│              所有曝光 (Impression)                │
│   ┌─────────────────────────────────┐            │
│   │        点击 (Click)              │            │
│   │   ┌─────────────────┐           │            │
│   │   │  转化 (Conversion)│           │            │
│   │   └─────────────────┘           │            │
│   └─────────────────────────────────┘            │
│                                                  │
│  训练空间 = 只有点击样本 (很小!)                      │
│  推理空间 = 全部曝光样本 (很大!)                      │
└─────────────────────────────────────────────────┘
```

### 问题1: 样本选择偏差 (Sample Selection Bias, SSB)

- 训练只用**点击样本**，推理却要在**全空间**做预测，分布不一致
- 点击行为本身就是一个筛选条件，被点击的样本天然偏向"用户感兴趣的"
- 模型在有偏子集上学，在全集上推理，泛化性差

**类比**: 用北京的气温数据训练模型，拿到全国去预测 —— 不具备代表性。

### 问题2: 数据稀疏 (Data Sparsity, DS)

- CTR 训练样本 = 全部曝光（如 89.5 亿）
- CVR 训练样本 = 仅点击（如 3.24 亿，仅约 4%）
- Embedding 层参数量巨大，4% 的数据喂不饱

---

## 二、核心思想

利用用户行为的**链式关系**:

```
impression → click → conversion
```

推导出一个等式：

```
pCTCVR = pCTR × pCVR

即: p(点击且转化|曝光) = p(点击|曝光) × p(转化|点击,曝光)
```

**不直接建模 pCVR**，而是把它当作中间变量，通过同时建模 pCTR 和 pCTCVR 来隐式约束 pCVR。

---

## 三、模型架构

```
        输入特征 (user field, item field, ...)
        ┌──────────────┬──────────────┐
        │              │              │
        ▼              │              ▼
  ┌───────────┐   Shared Embedding   ┌───────────┐
  │ Embedding │◄═══(参数共享)═══════►│ Embedding │
  │   Layer   │   Lookup Table       │   Layer   │
  └─────┬─────┘                      └─────┬─────┘
        │                                  │
        ▼                                  ▼
  ┌───────────┐                      ┌───────────┐
  │ Field-wise│                      │ Field-wise│
  │  Pooling  │                      │  Pooling  │
  └─────┬─────┘                      └─────┬─────┘
        │                                  │
        ▼                                  ▼
  ┌───────────┐                      ┌───────────┐
  │Concatenate│                      │Concatenate│
  └─────┬─────┘                      └─────┬─────┘
        │                                  │
        ▼                                  ▼
  ┌───────────┐                      ┌───────────┐
  │    MLP    │                      │    MLP    │
  │ 360→200   │                      │ 360→200   │
  │ →80→2     │                      │ →80→2     │
  └─────┬─────┘                      └─────┬─────┘
        │                                  │
        ▼                                  ▼
     [pCVR]                             [pCTR]
        │                                  │
        └──────────┐    ┌──────────────────┘
                   ▼    ▼
                 ┌────────┐
                 │   ×    │  (element-wise 相乘)
                 └───┬────┘
                     ▼
                  [pCTCVR]

   ◄── CVR 子网络 ──►   ◄── CTR 子网络 ──►
      (主任务)              (辅助任务)
```

### 架构要点

| 组件 | 说明 |
|------|------|
| Shared Embedding | CTR 和 CVR 网络共用同一张 Embedding 查找表 |
| CTR Tower | 预测 pCTR = p(点击\|曝光)，在全空间训练 |
| CVR Tower | 预测 pCVR = p(转化\|点击,曝光)，无直接 loss |
| 乘法层 | pCTCVR = pCTR × pCVR，输出保证在 [0,1] |

---

## 四、两个关键设计

### 4.1 全空间建模 — 解决 SSB

```
传统方法:                       ESMM:

训练: 点击样本 → pCVR           训练: 全部曝光 → pCTR
推理: 全部曝光 → pCVR                 全部曝光 → pCTCVR
      ↑ 分布不匹配!                   pCVR 通过乘法隐式推出
                                      ↑ 全空间一致!
```

pCTR 和 pCTCVR 都在全部曝光样本上训练，pCVR 作为中间变量自然也在全空间上有效。

### 4.2 Embedding 共享 — 解决 DS

```
CTR 样本量:  ████████████████████████  (全部曝光, 89.5亿)
CVR 样本量:  █                         (仅点击, 3.24亿, 约4%)

共享 Embedding → CVR 网络借助 CTR 海量样本学到更好的特征表示
```

**梯度流动**:

```
                Embedding (共享)     CTR MLP      CVR MLP
                ──────────────      ────────     ────────
CTR loss 梯度:       ✓                 ✓            ✗
CTCVR loss 梯度:     ✓                 ✓            ✓
```

同一张 Embedding 表同时接收两路梯度，CTR 海量数据帮 Embedding 学到更好的表示，CVR Tower 直接受益。

---

## 五、损失函数

```
L = L_ctr + L_ctcvr

L_ctr   = Σ CrossEntropy(y_i, f_ctr(x_i))             ← 点击标签，全空间
L_ctcvr = Σ CrossEntropy(y_i & z_i, f_ctr × f_cvr)    ← 点击且转化标签，全空间

注意: 没有单独的 L_cvr！CVR 通过乘法约束隐式学习。
```

---

## 六、PyTorch 实现

```python
import torch
import torch.nn as nn


class ESMM(nn.Module):
    def __init__(self, feature_dims, embed_dim=18, mlp_dims=[360, 200, 80]):
        super().__init__()

        # ===== Shared Embedding Layer =====
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in feature_dims.items()
        })

        input_dim = len(feature_dims) * embed_dim

        # ===== CTR Tower =====
        self.ctr_mlp = self._build_mlp(input_dim, mlp_dims)

        # ===== CVR Tower =====
        self.cvr_mlp = self._build_mlp(input_dim, mlp_dims)

    def _build_mlp(self, input_dim, mlp_dims):
        layers = []
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, features):
        # Shared Embedding + Concatenate
        emb_list = [self.embeddings[name](features[name])
                    for name in features]
        x = torch.cat(emb_list, dim=-1)

        # 两个 Tower 各自过 MLP
        pCTR = self.ctr_mlp(x)
        pCVR = self.cvr_mlp(x)

        # 核心: element-wise 相乘
        pCTCVR = pCTR * pCVR

        return pCTR, pCVR, pCTCVR


def esmm_loss(pCTR, pCTCVR, click_label, conversion_label):
    bce = nn.BCELoss()
    loss_ctr = bce(pCTR, click_label)
    ctcvr_label = click_label * conversion_label   # y & z
    loss_ctcvr = bce(pCTCVR, ctcvr_label)
    return loss_ctr + loss_ctcvr                   # 没有 loss_cvr
```

### 代码与论文对应关系

```
论文概念                    代码位置
──────────────────────────────────────────────────
Shared Lookup Table     →  self.embeddings (两个Tower共用同一个)
Embedding Layer         →  self.embeddings[name](features[name])
Field-wise Pooling      →  每个域一个embedding，直接取出
Concatenate             →  torch.cat(emb_list, dim=-1)
MLP                     →  self.ctr_mlp / self.cvr_mlp
element-wise ×          →  pCTR * pCVR
L_ctr + L_ctcvr         →  esmm_loss()
```

---

## 七、实验效果

| Model | CVR Task AUC | CTCVR Task AUC |
|-------|-------------|----------------|
| BASE | 66.00 | 62.07 |
| AMAN | 65.21 | 63.53 |
| OVERSAMPLING | 67.18 | 63.05 |
| UNBIAS | 66.65 | 63.56 |
| DIVISION | 67.56 | 63.62 |
| ESMM-NS (不共享Embedding) | 68.25 | 64.44 |
| **ESMM** | **68.56** | **65.32** |

vs BASE: CVR +2.56%, CTCVR +3.25%（工业界 +0.1% 就很显著）

---

## 八、ESMM 的局限与误差传播

### 乘法带来的误差传播

```
pCTCVR = pCTR × pCVR

pCVR 的相对误差 ≈ pCTCVR 的相对误差 + pCTR 的相对误差
```

CTR 预估偏了 → pCVR 被迫偏移来补偿 → **CTR 的误差"感染"CVR**。

#### 具体例子

```
真实: pCTR=0.04, pCVR=0.05, pCTCVR=0.002

若 pCTR 被高估为 0.06 (+50%):
  0.002 = 0.06 × pCVR → pCVR ≈ 0.033 (偏低 34%!)
```

#### 根本原因

```
ESMM 的 Loss = L_ctr + L_ctcvr （没有 L_cvr）

pCVR 没有自己的直接监督信号
它完全靠 pCTCVR 的 loss 通过乘法间接学习
```

#### 容易暴露问题的场景

- **冷启动**: 新商品 pCTR 不准 → pCVR 跟着偏
- **大促流量**: pCTR 普遍升高 → pCVR 被系统性压低
- **异常流量**: 机器人点击等

### 后续改进方向

| 方法 | 思路 | 做法 |
|------|------|------|
| ESCM² (阿里 2021) | 给 CVR 加直接监督 | L = L_ctr + L_ctcvr + λ·L_cvr (用 IPW 纠偏) |
| 解耦结构 | 打破乘法硬约束 | CTCVR 独立 Tower + 软正则 ‖pCTCVR - pCTR×pCVR‖² |
| Stop Gradient | 截断梯度传播 | `pCTCVR = pCTR.detach() * pCVR` |

---

## 九、CTR/CVR 联合建模演进

```
2018  ESMM (阿里)     ← 开山之作，乘法链式分解
  │
2018  MMoE (Google)   ← 通用多任务学习，多Expert+Gate软共享
  │
2019  ESM² (阿里)     ← ESMM升级，多步序列建模 (曝光→点击→加购→支付)
  │
2020  PLE (腾讯)      ← 解决任务间跷跷板问题，区分共享/专有 Expert
  │
2021  ESCM² (阿里)    ← 因果推断视角纠偏，给 CVR 加直接监督
```

### 如何选择？

| 场景 | 推荐 |
|------|------|
| 刚起步，快速上线 | ESMM（简单有效） |
| 任务多且相关性不确定 | MMoE |
| 明确观察到跷跷板现象 | PLE |
| 业务链路长 (曝光→点击→加购→支付) | ESM² |
| 对纠偏要求严格 | ESCM² |

---

## 十、一句话总结

> ESMM 利用 **impression → click → conversion** 的链式关系，通过 **pCTCVR = pCTR × pCVR** 把 CVR 建模从"点击子空间"拉到"全曝光空间"，同时用 Embedding 共享从 CTR 海量样本中迁移特征表示，一举解决样本选择偏差和数据稀疏两大问题。代价是 CVR 无直接监督，存在误差传播风险。
