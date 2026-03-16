# KAD-Disformer 改造清单：模型内可学习分解 + 重构

本文档给出“从固定小波去噪监督”迁移到“模型内可学习频域分解 + 重构监督”的具体改造点。

## 改造目标

当前训练目标：

- `loss = MSE(output, denoised_seq)`（`denoised_seq` 来自数据集的小波去噪）

改造后目标：

- `loss = L_recon + λ1 * L_overlap + λ2 * L_cover`
- `L_recon = MSE(output, seq_context)`（直接重构原始输入）
- `L_overlap` / `L_cover` 由模型内部可学习频段掩码产生

---

## 1) `src/utils/dataset.py`

### 修改点 A：`KAD_DisformerTrainSet` 输出从 3 项改为 2 项

当前：

- `__getitem__` 返回 `(context_flow, history_flow, denoised_context)`

目标：

- `__getitem__` 返回 `(context_flow, history_flow)`

### 建议改法

1. 在 `KAD_DisformerTrainSet.__init__` 中删除（或保留但不使用）：
   - `self.denoised_context = DenoisedUTSDataset(...)`
2. 在 `KAD_DisformerTrainSet.__getitem__` 改为：

```python
def __getitem__(self, index):
    return self.context_flow[index], self.history_flow[index]
```

> 说明：`DenoisedUTSDataset` 类可暂时保留，避免影响其它实验脚本；但训练主线不再依赖它。

---

## 2) `src/models/KAD_Disformer.py`

### 修改点 B：新增“可学习频域分解模块”

新增类（建议放在本文件顶部）：

- `LearnableFreqDecomp`

输入输出建议：

- 输入：`x`，形状 `[B, S, W]`（batch, seq_len, win_len）
- 输出：
  - `bands`: `[B, K, S, W]`（K 个频段的时域分量）
  - `mask`: `[K, F]`（频域掩码，F 为 rFFT 频点数）

核心逻辑建议：

1. `X = torch.fft.rfft(x, dim=-1)`，得到 `[B, S, F]`
2. 用可训练参数生成 `K` 个软掩码（建议 `sigmoid` 后归一化）：
   - `raw_mask` 参数形状 `[K, F]`
   - `mask = sigmoid(raw_mask)`
   - `mask = mask / (mask.sum(dim=0, keepdim=True) + 1e-6)`
3. `X_k = X * mask[k]`
4. `x_k = irfft(X_k, n=W, dim=-1)`，拼成 `bands`

### 修改点 C：在 `KAD_Disformer` 中接入分解与融合

建议新增成员：

- `self.freq_decomp = LearnableFreqDecomp(num_bands=3, win_len=patch_size)`
- `self.band_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))`

建议前向流程：

1. `seq_context` 和 `seq_history` 先过 `freq_decomp` 得到各频段分量
2. 每个频段进入原有 `context_block/history_block`（先复用同一套 block，低风险）
3. 每频段得到一个重构结果 `y_k`
4. 通过门控融合得到最终输出 `y`
5. 若 `return_aux=True`，返回正则项需要的中间信息：
   - `mask`
   - `band_weights`（门控权重）

建议接口：

```python
def forward(self, seq_context, seq_history, return_aux: bool = False):
    ...
    if return_aux:
        return y, {"mask": mask, "band_weights": g}
    return y
```

---

## 3) `src/train.py`

### 修改点 D：训练循环适配新 dataset 输出与新 loss

当前循环解包：

- `for ..., (seq_context, seq_history, denoised_seq) in ...`

改为：

- `for ..., (seq_context, seq_history) in ...`

前向改为：

```python
output, aux = model(seq_context, seq_history, return_aux=True)
```

loss 改为：

```python
recon_loss = criterion(output, seq_context)

mask = aux["mask"]  # [K, F]
overlap = ((mask.unsqueeze(0) * mask.unsqueeze(1)).sum(dim=-1)).sum() - (mask * mask).sum()
cover = ((mask.sum(dim=0) - 1.0) ** 2).mean()

loss = recon_loss + args.lambda_overlap * overlap + args.lambda_cover * cover
```

### 修改点 E：新增命令行参数

在 `parse_args()` 中增加：

- `--lambda_overlap`，默认 `1e-3`
- `--lambda_cover`，默认 `1e-3`

---

## 4) `src/finetune.py`

### 修改点 F：微调训练与评估保持一致

微调训练环节做和 `train.py` 一样的改动：

1. dataloader 解包改为 2 项
2. `model(..., return_aux=True)`
3. 损失替换为 `recon + overlap + cover`
4. 新增 `lambda` 参数

评估阶段可保持现有流程（重构误差打分）不变。

---

## 5) `src/test.py`

### 修改点 G：仅兼容新 forward 签名

如果 `forward` 默认 `return_aux=False`，这里可不改。

若你改成必须返回 tuple，则改为：

```python
output, _ = model(seq_context, seq_history, return_aux=True)
```

---

## 6) 推荐新增文件（可选）

### `src/models/Frequency.py`（可选）

如果不想让 `KAD_Disformer.py` 过长，建议把 `LearnableFreqDecomp` 和正则函数放到新文件中。

可提供函数：

- `build_soft_masks(raw_mask)`
- `compute_freq_regularization(mask)`

---

## 7) 最小可跑版本（建议先做）

1. `num_bands=3`（低/中/高）
2. 先共享原有 block，不新增额外 backbone 参数
3. 先启用：
   - `L_recon`
   - `L_overlap`（`1e-3`）
4. `L_cover` 可第二轮加

---

## 8) 验证清单

1. 训练脚本能正常启动，无 shape 报错
2. `loss` 能稳定下降
3. `test.py` 能输出 F1
4. 对比基线：
   - 原版（fixed denoise target）
   - 新版（learnable decomp）
5. 若新版 F1 上升，再继续做“分频异常分数”可解释化输出

---

## 9) 风险提示（实现时注意）

1. `mask` 正则不要过大，否则会压制重构能力
2. `irfft` 必须指定 `n=win_len`，避免长度漂移
3. 保证 `mask.sum(dim=0)` 接近 1，避免某些频点无人负责
4. 先不引入空间模块（graph/Mamba），当前单 KPI 场景优先把分解学好

