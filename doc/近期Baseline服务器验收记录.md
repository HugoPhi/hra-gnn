# 近期 Baseline 服务器验收记录

更新时间：2026-07-02

## 1. 服务器环境

```text
GPU: Tesla V100-PCIE-16GB
可见 GPU 数: 1
PyTorch: 2.5.1+cu121
PyG: 2.6.1
torch-scatter: 2.1.2+pt25cu121
```

服务器项目 37 项测试全部通过。官方 baseline 仓库由
`configs/baselines.lock.yaml` 锁定 commit，并下载到不纳入 Git 的 `external/`。

## 2. Smoke 验收目的

本轮只验证：

- 官方数据能正确加载；
- 官方核心模型能完成前向和反向传播；
- loss 能下降；
- 能产生有限的图级异常分数和 AUROC；
- CUDA、PyG 和扩展依赖兼容。

短 epoch 结果不能作为论文最终对比成绩。

## 3. 结果

| 模型 | 官方数据 | 试验预算 | Smoke 结果 | 状态 |
|---|---|---|---:|---|
| SIGNET | AIDS | 1 trial，1 epoch | AUROC 0.7235 | PASS |
| CVTGAD | BZR | 2 folds，10 epochs | AUROC 0.6728 +/- 0.0041 | PASS |
| MUSE | BZR | 1 trial，表示/分类器各 1 epoch | AUROC 0.6250，AP 0.4635 | PASS |
| GLADMamba | BZR | 2 folds，10 epochs | AUROC 0.6165 +/- 0.0086 | PASS |

所有结果均来自服务器 V100，不是本地 CPU 模拟值。

## 4. 发现的官方工程问题

### 4.1 CVTGAD

官方 `main.py` 强制执行：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

服务器只有 GPU 0，导致官方代码报 `No CUDA GPUs are available`。项目 patch 改为
`setdefault`，允许外部指定 GPU，不改变模型、损失或分数。

### 4.2 MUSE

官方代码使用旧 PyG 导入路径：

```python
from torch_geometric.utils.subgraph import subgraph
```

PyG 2.6 已移除该模块路径。兼容 patch 改为从 `torch_geometric.utils` 导入同名函数。

官方入口还固定执行很大的网格：

- 5 次试验；
- 10 个表示学习 checkpoint；
- 3 种分类器维度；
- 3 种学习率；
- 每个分类器 500 epoch。

运行控制 patch 将这些预算暴露为参数，默认值仍与官方一致。

### 4.3 官方协议的数据使用问题

- MUSE 用带异常标签的 validation AUROC early stopping 和选择超参数；
- CVTGAD 训练期间反复计算 test AUROC；
- GLADMamba 训练期间反复计算 test AUROC，并跨 trial 选择平均 test AUROC 最好的 epoch。

因此本轮只标记为“官方实现能运行”。论文公平表必须使用独立 `fair` runner。

## 5. 下一步验收

每个模型还需完成：

1. 使用统一图表示或 TU Dataset 导出；
2. 使用冻结的 train/validation/test 图 ID；
3. 训练只读取正常训练图；
4. checkpoint 不读取测试标签；
5. 输出 AUROC、AP、预算指标、阈值指标和效率指标；
6. 在 TraceLog、FlowGraph、HDFS、ADFA-LD 上运行五个 seed。

## 6. Fair Runner 第一轮验收

CVTGAD 和 GLADMamba 已完成公平 runner：

- 使用项目冻结 split；
- 训练只读取正常图；
- 固定 epoch，不根据测试 AUC 选择 checkpoint；
- 测试标签只在最终指标计算时读取；
- 阈值来自正常训练分数；
- 输出 AUROC、AP、预算指标、F1/MCC、参数量和显存。

### 6.1 TraceLog，128 图/split，1 epoch

| 模型 | AUROC | AP | 峰值显存 |
|---|---:|---:|---:|
| CVTGAD-fair | 0.5435 | 0.5982 | 2523.7 MB |
| GLADMamba-fair | 0.5803 | 0.5188 | 2061.0 MB |

### 6.2 FlowGraph，32 图/split，1 epoch，seed 11

| 模型 | AUROC | AP | F1 | 峰值显存 |
|---|---:|---:|---:|---:|
| CVTGAD-fair | 0.5469 | 0.4934 | 0.0833 | 8860.9 MB |
| GLADMamba-fair | 0.7656 | 0.6565 | 0.7317 | 6191.9 MB |

以上仍是链路测试，不是最终论文数值。

### 6.3 FlowGraph 的方法约束

CVTGAD 的节点交叉注意力对 batch 中全部节点形成稠密矩阵：

- `batch_size=32` 尝试申请约 257 GB 显存并 OOM；
- `batch_size=1` 没有图级对比负样本，官方损失必然除零；
- `batch_size=2` 可运行，但单张 V100 上接近显存上限。

此外，两种官方实现均未处理零范数嵌入，FlowGraph 会产生 `NaN/Inf`。公平 runner
采用带 epsilon 的等价归一化，并在结果中标记 `numerically_stabilized=true`。

## 7. 四模型统一 Smoke Matrix

统一调度器在服务器完成 8 个组合，其中 7 个完成、1 个按预期记录为不可行：

| 数据集 | 模型 | AUROC | AP | 峰值显存 | 状态 |
|---|---|---:|---:|---:|---|
| TraceLog | SIGNET-fair | 0.4746 | 0.4961 | 40.4 MB | COMPLETE |
| TraceLog | CVTGAD-fair | 0.5139 | 0.5143 | 2561.8 MB | COMPLETE |
| TraceLog | MUSE-fair | 0.3867 | 0.5066 | 72.2 MB | COMPLETE |
| TraceLog | GLADMamba-fair | 0.6206 | 0.5503 | 1882.5 MB | COMPLETE |
| FlowGraph | SIGNET-fair | 0.1875 | 0.3609 | 1143.6 MB | COMPLETE |
| FlowGraph | CVTGAD-fair | 0.5469 | 0.4934 | 8860.9 MB | COMPLETE |
| FlowGraph | MUSE-fair | -- | -- | -- | N/A：8302 节点超过稠密邻接限制 |
| FlowGraph | GLADMamba-fair | 0.7656 | 0.6565 | 6191.9 MB | COMPLETE |

这些均为 1 epoch、小样本链路验收，不能作为论文性能结论。它们的作用是证明统一矩阵、
失败记录和 LaTeX 汇总链路已经贯通。
