"""
DIN & DIEN 模型实现 (PyTorch)
=============================
DIN:  Deep Interest Network (Alibaba, 2018)
DIEN: Deep Interest Evolution Network (Alibaba, 2019)

包含: 模型定义、模拟数据生成、训练、推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

# ============================================================
# 通用组件
# ============================================================

class Dice(nn.Module):
    """DIN 论文提出的数据自适应激活函数

    核心思想: 根据数据分布动态调整整流点(而非PReLU的硬阈值0)
    p_i = sigmoid((y_i - E[y_i]) / sqrt(Var[y_i] + eps))
    f(y_i) = a_i * (1 - p_i) * y_i + p_i * y_i
    """
    def __init__(self, dim: int, eps: float = 1e-9):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))  # 左侧斜率
        self.bn = nn.BatchNorm1d(dim, eps=eps, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 3D 输入 (B, T, D): 先reshape成2D再还原
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(-1, D)
            p = torch.sigmoid(self.bn(x_flat)).reshape(B, T, D)
        else:
            p = torch.sigmoid(self.bn(x))
        return self.alpha * (1 - p) * x + p * x


class AttentionUnit(nn.Module):
    """DIN 的注意力激活单元

    计算每个历史行为与候选广告之间的相关性权重
    输入: 候选广告embedding, 历史行为embedding序列
    输出: 加权后的用户兴趣表示
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 输入: [行为embedding, 候选embedding, 两者差, 两者乘积]
        input_dim = embed_dim * 4
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Dice(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate: torch.Tensor,      # (B, E)
        behaviors: torch.Tensor,       # (B, T, E)
        behavior_mask: torch.Tensor,   # (B, T) bool, True=有效
    ) -> torch.Tensor:                 # (B, E) 加权兴趣表示
        B, T, E = behaviors.shape
        # 扩展候选广告到序列长度
        cand = candidate.unsqueeze(1).expand(-1, T, -1)  # (B, T, E)
        # 拼接4种交互特征
        interaction = torch.cat([
            behaviors, cand,
            behaviors - cand,
            behaviors * cand,
        ], dim=-1)  # (B, T, 4E)
        # 计算注意力分数
        scores = self.fc(interaction).squeeze(-1)  # (B, T)
        # mask 掉padding位置
        scores = scores.masked_fill(~behavior_mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        # 加权求和
        output = torch.bmm(weights.unsqueeze(1), behaviors).squeeze(1)  # (B, E)
        return output


# ============================================================
# DIN 模型
# ============================================================

class DIN(nn.Module):
    """Deep Interest Network

    结构: Embedding → Attention加权 → 拼接其他特征 → MLP → CTR
    """
    def __init__(
        self,
        num_items: int,
        num_cats: int,
        embed_dim: int = 16,
        mlp_dims: list = None,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64]

        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_cats, embed_dim, padding_idx=0)

        # 行为embedding = item_emb + cat_emb, 拼接后维度 = 2*embed_dim
        behavior_dim = 2 * embed_dim
        self.attention = AttentionUnit(behavior_dim)

        # MLP 输入: 用户兴趣(behavior_dim) + 候选广告(behavior_dim)
        mlp_input_dim = behavior_dim * 2
        layers = []
        for dim in mlp_dims:
            layers.extend([nn.Linear(mlp_input_dim, dim), Dice(dim)])
            mlp_input_dim = dim
        layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hist_item_ids: torch.Tensor,   # (B, T)
        hist_cat_ids: torch.Tensor,     # (B, T)
        cand_item_id: torch.Tensor,     # (B,)
        cand_cat_id: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        # mask: padding位置为0
        mask = hist_item_ids > 0  # (B, T)

        # Embedding
        hist_emb = torch.cat([
            self.item_emb(hist_item_ids),
            self.cat_emb(hist_cat_ids),
        ], dim=-1)  # (B, T, 2E)
        cand_emb = torch.cat([
            self.item_emb(cand_item_id),
            self.cat_emb(cand_cat_id),
        ], dim=-1)  # (B, 2E)

        # Attention加权的用户兴趣
        user_interest = self.attention(cand_emb, hist_emb, mask)  # (B, 2E)

        # 拼接 → MLP
        concat = torch.cat([user_interest, cand_emb], dim=-1)
        logit = self.mlp(concat).squeeze(-1)
        return logit


# ============================================================
# DIEN 模型组件
# ============================================================

class AUGRU(nn.Module):
    """GRU with Attentional Update Gate (AUGRU)

    DIEN 的核心创新: 将注意力机制嵌入 GRU 的 update gate
    ũ'_t = a_t * u'_t     (注意力缩放 update gate)
    h'_t = (1 - ũ'_t) ⊙ h'_{t-1} + ũ'_t ⊙ h̃'_t

    效果: 与目标商品无关的兴趣被"跳过", 相关兴趣主导演化方向
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Update gate
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size, bias=False)
        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,        # (B, T, input_size)
        att_scores: torch.Tensor,     # (B, T) 注意力分数
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:               # (B, hidden_size) 最终隐状态
        B, T, _ = inputs.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=inputs.device)

        h = h0
        for t in range(T):
            x_t = inputs[:, t, :]        # (B, input_size)
            a_t = att_scores[:, t].unsqueeze(-1)  # (B, 1)

            # 标准 GRU 计算
            u_t = torch.sigmoid(self.W_u(x_t) + self.U_u(h))  # update gate
            r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h))  # reset gate
            h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h))  # candidate

            # AUGRU 的关键: 注意力缩放 update gate
            u_t_prime = a_t * u_t  # ũ'_t = a_t * u'_t

            h = (1 - u_t_prime) * h + u_t_prime * h_tilde

        return h  # 最终兴趣状态


class DIEN(nn.Module):
    """Deep Interest Evolution Network

    结构:
      行为序列 → Embedding → GRU(兴趣抽取层) + 辅助损失
                           → AUGRU(兴趣演化层, 用候选广告计算注意力)
                           → 拼接其他特征 → MLP → CTR
    """
    def __init__(
        self,
        num_items: int,
        num_cats: int,
        embed_dim: int = 16,
        gru_hidden: int = 32,
        mlp_dims: list = None,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64]

        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_cats, embed_dim, padding_idx=0)

        behavior_dim = 2 * embed_dim  # item_emb + cat_emb 拼接
        self.gru_hidden = gru_hidden

        # ---- Interest Extractor Layer (兴趣抽取层) ----
        self.gru = nn.GRU(behavior_dim, gru_hidden, batch_first=True)

        # ---- 辅助损失的分类头 ----
        # 用 h_t 预测下一个行为 b_{t+1}: sigmoid(h_t · e_{t+1})
        # (不需要额外参数, 直接做内积)

        # ---- Interest Evolving Layer (兴趣演化层) ----
        # 注意力: 计算每个兴趣状态 h_t 与候选广告的相关性
        self.attention_fc = nn.Sequential(
            nn.Linear(gru_hidden + behavior_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.augru = AUGRU(gru_hidden, gru_hidden)

        # ---- MLP ----
        mlp_input_dim = gru_hidden + behavior_dim  # 兴趣演化输出 + 候选广告
        layers = []
        for dim in mlp_dims:
            layers.extend([nn.Linear(mlp_input_dim, dim), Dice(dim)])
            mlp_input_dim = dim
        layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _compute_attention(
        self,
        gru_states: torch.Tensor,  # (B, T, H)
        cand_emb: torch.Tensor,    # (B, behavior_dim)
    ) -> torch.Tensor:             # (B, T)
        B, T, H = gru_states.shape
        cand_exp = cand_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, behavior_dim)
        concat = torch.cat([gru_states, cand_exp], dim=-1)  # (B, T, H+behavior_dim)
        scores = self.attention_fc(concat).squeeze(-1)       # (B, T)
        return torch.sigmoid(scores)

    def forward(
        self,
        hist_item_ids: torch.Tensor,   # (B, T)
        hist_cat_ids: torch.Tensor,    # (B, T)
        cand_item_id: torch.Tensor,    # (B,)
        cand_cat_id: torch.Tensor,     # (B,)
        neg_item_ids: Optional[torch.Tensor] = None,  # (B, T) 辅助损失负样本
        neg_cat_ids: Optional[torch.Tensor] = None,    # (B, T)
    ):
        """
        返回:
          logit: (B,) CTR预测
          aux_loss: 辅助损失 (训练时), 推理时为0
        """
        mask = hist_item_ids > 0  # (B, T)

        # ---- Embedding ----
        hist_emb = torch.cat([
            self.item_emb(hist_item_ids),
            self.cat_emb(hist_cat_ids),
        ], dim=-1)  # (B, T, 2E)
        cand_emb = torch.cat([
            self.item_emb(cand_item_id),
            self.cat_emb(cand_cat_id),
        ], dim=-1)  # (B, 2E)

        # ---- Interest Extractor: GRU ----
        gru_out, _ = self.gru(hist_emb)  # (B, T, H)

        # ---- Auxiliary Loss (辅助损失) ----
        aux_loss = torch.tensor(0.0, device=hist_emb.device)
        if neg_item_ids is not None and self.training:
            neg_emb = torch.cat([
                self.item_emb(neg_item_ids),
                self.cat_emb(neg_cat_ids),
            ], dim=-1)  # (B, T, 2E)

            # h_t 预测 b_{t+1}: 正样本用 hist_emb[:, 1:], 负样本用 neg_emb[:, 1:]
            # GRU隐状态需要投影到embedding空间
            h_states = gru_out[:, :-1, :]       # (B, T-1, H)
            pos_next = hist_emb[:, 1:, :]       # (B, T-1, 2E)
            neg_next = neg_emb[:, 1:, :]        # (B, T-1, 2E)
            valid = mask[:, 1:]                 # (B, T-1)

            # 简化: 用内积后sigmoid计算概率 (h_t 线性投影到 behavior_dim)
            # 这里直接用前 behavior_dim 维做内积（与论文一致的简化实现）
            H = h_states.shape[-1]
            E2 = pos_next.shape[-1]
            # 如果维度不同, 投影对齐
            if H != E2:
                if not hasattr(self, '_aux_proj'):
                    self._aux_proj = nn.Linear(H, E2, bias=False).to(h_states.device)
                h_proj = self._aux_proj(h_states)
            else:
                h_proj = h_states

            pos_logits = (h_proj * pos_next).sum(dim=-1)  # (B, T-1)
            neg_logits = (h_proj * neg_next).sum(dim=-1)  # (B, T-1)

            pos_loss = -F.logsigmoid(pos_logits)
            neg_loss = -F.logsigmoid(-neg_logits)
            aux_loss = ((pos_loss + neg_loss) * valid.float()).sum() / valid.float().sum().clamp(min=1)

        # ---- Interest Evolving: AUGRU ----
        att_scores = self._compute_attention(gru_out, cand_emb)  # (B, T)
        att_scores = att_scores * mask.float()  # mask padding

        interest = self.augru(gru_out, att_scores)  # (B, H)

        # ---- MLP ----
        concat = torch.cat([interest, cand_emb], dim=-1)
        logit = self.mlp(concat).squeeze(-1)

        return logit, aux_loss


# ============================================================
# 模拟数据集
# ============================================================

class CTRDataset(Dataset):
    """模拟电商CTR数据

    每条样本: 用户历史行为序列 + 候选商品 + 标签(是否点击)
    """
    def __init__(
        self,
        num_samples: int = 10000,
        num_items: int = 1000,
        num_cats: int = 50,
        max_seq_len: int = 30,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.max_seq_len = max_seq_len

        self.hist_items = []
        self.hist_cats = []
        self.cand_items = []
        self.cand_cats = []
        self.neg_items = []   # DIEN辅助损失用
        self.neg_cats = []
        self.labels = []

        for _ in range(num_samples):
            # 随机生成行为序列长度
            seq_len = rng.randint(5, max_seq_len + 1)
            # 随机生成类目(模拟用户兴趣集中在几个类目)
            user_cats = rng.choice(range(1, num_cats), size=3, replace=False)

            # 生成历史行为
            h_cats = rng.choice(user_cats, size=seq_len)
            h_items = rng.randint(1, num_items, size=seq_len)
            # padding到固定长度
            pad_len = max_seq_len - seq_len
            h_items_pad = np.pad(h_items, (0, pad_len), constant_values=0)
            h_cats_pad = np.pad(h_cats, (0, pad_len), constant_values=0)

            # 候选商品
            cand_item = rng.randint(1, num_items)
            # 50%概率候选商品的类目在用户兴趣类目中(模拟正样本)
            if rng.random() > 0.5:
                cand_cat = rng.choice(user_cats)
                label = 1
            else:
                cand_cat = rng.randint(1, num_cats)
                label = 0

            # 负样本(DIEN辅助损失用): 随机采样非点击商品
            n_items = rng.randint(1, num_items, size=max_seq_len)
            n_cats = rng.randint(1, num_cats, size=max_seq_len)
            n_items[seq_len:] = 0
            n_cats[seq_len:] = 0

            self.hist_items.append(h_items_pad)
            self.hist_cats.append(h_cats_pad)
            self.cand_items.append(cand_item)
            self.cand_cats.append(cand_cat)
            self.neg_items.append(n_items)
            self.neg_cats.append(n_cats)
            self.labels.append(label)

        self.hist_items = np.array(self.hist_items)
        self.hist_cats = np.array(self.hist_cats)
        self.cand_items = np.array(self.cand_items)
        self.cand_cats = np.array(self.cand_cats)
        self.neg_items = np.array(self.neg_items)
        self.neg_cats = np.array(self.neg_cats)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'hist_items': torch.LongTensor(self.hist_items[idx]),
            'hist_cats': torch.LongTensor(self.hist_cats[idx]),
            'cand_item': torch.LongTensor([self.cand_items[idx]]).squeeze(),
            'cand_cat': torch.LongTensor([self.cand_cats[idx]]).squeeze(),
            'neg_items': torch.LongTensor(self.neg_items[idx]),
            'neg_cats': torch.LongTensor(self.neg_cats[idx]),
            'label': torch.FloatTensor([self.labels[idx]]).squeeze(),
        }


# ============================================================
# 训练与推理
# ============================================================

def train_model(model, train_loader, val_loader, model_name: str, epochs: int = 5,
                lr: float = 1e-3, aux_loss_weight: float = 0.5):
    """统一训练函数, 同时支持 DIN 和 DIEN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_dien = isinstance(model, DIEN)

    print(f"\n{'='*60}")
    print(f" 训练 {model_name}")
    print(f" Device: {device} | Epochs: {epochs} | LR: {lr}")
    params = sum(p.numel() for p in model.parameters())
    print(f" 参数量: {params:,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_aux = 0
        num_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if is_dien:
                logit, aux_loss = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                    batch['neg_items'], batch['neg_cats'],
                )
                ctr_loss = F.binary_cross_entropy_with_logits(logit, batch['label'])
                loss = ctr_loss + aux_loss_weight * aux_loss
                total_aux += aux_loss.item()
            else:
                logit = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                )
                loss = F.binary_cross_entropy_with_logits(logit, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 验证
        val_auc = evaluate(model, val_loader, device, is_dien)
        avg_loss = total_loss / num_batches
        aux_str = f" | Aux Loss: {total_aux / num_batches:.4f}" if is_dien else ""
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}{aux_str} | Val AUC: {val_auc:.4f}")

    return model


def evaluate(model, loader, device, is_dien: bool) -> float:
    """计算 AUC"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if is_dien:
                logit, _ = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                )
            else:
                logit = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                )
            preds = torch.sigmoid(logit)
            all_preds.append(preds.cpu())
            all_labels.append(batch['label'].cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return compute_auc(labels, preds)


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """手动计算 AUC (避免额外依赖sklearn)"""
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # 随机采样估算, 避免 O(n^2)
    n_samples = min(50000, len(pos) * len(neg))
    rng = np.random.RandomState(0)
    pos_samples = rng.choice(pos, size=n_samples, replace=True)
    neg_samples = rng.choice(neg, size=n_samples, replace=True)
    return float(np.mean(pos_samples > neg_samples) + 0.5 * np.mean(pos_samples == neg_samples))


def inference_demo(model, dataset, model_name: str, is_dien: bool, n: int = 5):
    """推理演示: 对几条样本做CTR预测"""
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'='*60}")
    print(f" {model_name} 推理演示")
    print(f"{'='*60}")

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

            if is_dien:
                logit, _ = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                )
            else:
                logit = model(
                    batch['hist_items'], batch['hist_cats'],
                    batch['cand_item'], batch['cand_cat'],
                )

            prob = torch.sigmoid(logit).item()
            label = sample['label'].item()
            seq_len = (sample['hist_items'] > 0).sum().item()
            cand = sample['cand_item'].item()

            print(f"  样本 {i}: 行为序列长度={seq_len}, 候选商品={cand}, "
                  f"预测CTR={prob:.4f}, 真实标签={int(label)}")


# ============================================================
# 主程序
# ============================================================

def main():
    # 超参数
    NUM_ITEMS = 1000
    NUM_CATS = 50
    EMBED_DIM = 16
    GRU_HIDDEN = 32
    MAX_SEQ_LEN = 30
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 1e-3

    print("生成模拟数据...")
    train_ds = CTRDataset(num_samples=10000, num_items=NUM_ITEMS, num_cats=NUM_CATS,
                          max_seq_len=MAX_SEQ_LEN, seed=42)
    val_ds = CTRDataset(num_samples=2000, num_items=NUM_ITEMS, num_cats=NUM_CATS,
                        max_seq_len=MAX_SEQ_LEN, seed=99)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"训练集: {len(train_ds)} 样本 | 验证集: {len(val_ds)} 样本")
    print(f"正样本比例: 训练={train_ds.labels.mean():.2%}, 验证={val_ds.labels.mean():.2%}")

    # ---- DIN ----
    din = DIN(num_items=NUM_ITEMS, num_cats=NUM_CATS, embed_dim=EMBED_DIM)
    din = train_model(din, train_loader, val_loader, model_name="DIN", epochs=EPOCHS, lr=LR)
    inference_demo(din, val_ds, "DIN", is_dien=False)

    # ---- DIEN ----
    dien = DIEN(num_items=NUM_ITEMS, num_cats=NUM_CATS, embed_dim=EMBED_DIM,
                gru_hidden=GRU_HIDDEN)
    dien = train_model(dien, train_loader, val_loader, model_name="DIEN", epochs=EPOCHS, lr=LR)
    inference_demo(dien, val_ds, "DIEN", is_dien=True)

    # ---- 保存模型 ----
    torch.save(din.state_dict(), '/tmp/din_model.pt')
    torch.save(dien.state_dict(), '/tmp/dien_model.pt')
    print(f"\n模型已保存到 /tmp/din_model.pt 和 /tmp/dien_model.pt")


if __name__ == '__main__':
    main()
