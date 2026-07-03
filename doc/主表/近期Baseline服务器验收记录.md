# 近期 Baseline 服务器验收记录

> 本文归档于 `doc/主表/`，保存近期 baseline 的环境、适配和服务器验收证据。

更新时间：2026-07-02

## 1. 服务器环境

```text
GPU: Tesla V100-PCIE-16GB
可见 GPU 数: 1
PyTorch: 2.5.1+cu121
PyG: 2.6.1
torch-scatter: 2.1.2+pt25cu121
```

服务器项目 41 项测试全部通过。官方 baseline 仓库由
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

下表是评分协议修正前的第一轮结果。SIGNET、CVTGAD 和 GLADMamba 的图级
对比损失使用当前测试 batch 内其他图作为负样本；TraceLog 和 FlowGraph 无法
像官方小数据集那样一次装入全部测试图，因此不同 batch 的分数不在统一尺度。
该表现已判定为**批次依赖的无效性能诊断**，只保留用于追踪问题，禁止引用。

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

### 7.1 修正方案

- 训练仍使用各方法原有的批内对比损失；
- 从正常训练图中固定抽取参考库；
- 每个测试图仅与同一正常参考库比较，测试图之间不互相充当负样本；
- 参考图与阈值校准图互斥，阈值仍只由正常训练数据产生；
- 截断子集改为按标签、随机种子分层随机抽样，不再取数据文件中的前 N 项；
- 结果记录正向 AUC、反向诊断 AUC、正常/异常分数中位数和评分协议；
- 反向 AUC 只用于发现语义错误，不用于事后翻转正式分数。

### 7.2 修正后的统一 Smoke Matrix

服务器 V100，seed 11，1 epoch；仍属于 `smoke_do_not_cite`：

| 数据集 | 模型 | AUROC | 反向诊断 AUROC | AP | 正常/异常分数中位数 |
|---|---|---:|---:|---:|---:|
| TraceLog | SIGNET-fair | 0.5713 | 0.4287 | 0.6118 | 4.1496 / 4.1585 |
| TraceLog | CVTGAD-fair | 0.5505 | 0.4495 | 0.5878 | 22.2437 / 22.6753 |
| TraceLog | MUSE-fair | 0.4219 | 0.5781 | 0.5441 | 232.0440 / 225.9413 |
| TraceLog | GLADMamba-fair | 0.5920 | 0.4080 | 0.5409 | 8.4745 / 8.7523 |
| FlowGraph | SIGNET-fair | 0.7949 | 0.2051 | 0.6565 | 3.4567 / 3.4807 |
| FlowGraph | CVTGAD-fair | 0.8086 | 0.1914 | 0.8382 | 3.1024 / 3.5023 |
| FlowGraph | MUSE-fair | -- | -- | -- | 稠密邻接超出 2048 节点限制 |
| FlowGraph | GLADMamba-fair | 0.4023 | 0.5977 | 0.4348 | 159.5584 / 158.9231 |

FlowGraph 的 SIGNET 从 0.1875 恢复到 0.7949，CVTGAD 从 0.5469 提升到
0.8086，证明旧协议确实存在严重的 batch 评分失真。TraceLog 仍接近随机水平，
需要通过更长训练判断是尚未收敛还是方法迁移能力有限。GLADMamba 在 FlowGraph
上的分数方向仍可疑，但不得根据测试反向 AUC 直接翻转。

### 7.3 证据等级与输出隔离

矩阵配置必须声明证据等级：

- `smoke_do_not_cite`：只验证链路，禁止进入论文表格；
- `diagnostic_not_final`：用于选择超参数范围和排查适配问题；
- 最终五随机种子完整数据实验才可标记为论文候选结果。

每个矩阵的模型结果写入各自的 `model_runs/`，不会再由相同的
`dataset/model/seed` 路径互相覆盖。

## 8. 当前诊断长跑

配置：`configs/experiments/fair_recent_diagnostic.yaml`。

- TraceLog：SIGNET、CVTGAD、MUSE、GLADMamba；
- FlowGraph：SIGNET、CVTGAD、GLADMamba，MUSE 因方法复杂度不适用；
- 预算：3 seeds，主要模型 20 epochs，扩大分层随机子集；
- 状态：2026-07-02 在服务器完成，21/21 组合成功；
- 用途：判断收敛趋势和迁移适用性，仍不作为最终论文结果。

### 8.1 三随机种子结果

| 数据集 | 模型 | AUROC | AP | 结论 |
|---|---|---:|---:|---|
| TraceLog | SIGNET-fair | 0.5052 +/- 0.0285 | 0.5365 +/- 0.0805 | 接近随机 |
| TraceLog | CVTGAD-fair | 0.4920 +/- 0.0654 | 0.5214 +/- 0.0521 | 接近随机 |
| TraceLog | MUSE-fair | 0.4880 +/- 0.0750 | 0.5078 +/- 0.0808 | 接近随机 |
| TraceLog | GLADMamba-fair | 0.5395 +/- 0.0476 | 0.5816 +/- 0.0474 | 略高于随机 |
| FlowGraph | SIGNET-fair | 0.5202 +/- 0.2812 | 0.5105 +/- 0.1503 | 极不稳定 |
| FlowGraph | CVTGAD-fair | 0.7227 +/- 0.2520 | 0.6807 +/- 0.2796 | 均值较高但极不稳定 |
| FlowGraph | GLADMamba-fair | 0.5104 +/- 0.2858 | 0.5779 +/- 0.2875 | 极不稳定 |

TraceLog 上 SIGNET 的训练 loss 从约 3.42 降至 2.10，但 AUROC 仍接近
0.5，说明链路和优化器正常，模型目标却未对齐该日志异常任务。FlowGraph
的单个 seed 可出现很高成绩，例如 CVTGAD seed 33 为 1.0，但另两个 seed
明显较低；不得选择最好 seed 作为论文结果。

### 8.2 阶段决策

- 不继续为这些跨域方法执行五 seed 全量盲跑；
- 将其保留为“近期通用图异常检测方法的公平迁移适配”补充实验；
- 主表算力优先投入 HRGCN、HRA-GNN 及相同任务协议下的基础方法；
- FlowGraph 必须同时报告统计基线，分析其容易被图规模等捷径分开的原因。

## 9. 直接任务基线

全量 TraceLog 实测每个 epoch 约 240 秒，单张 V100 上完成 30 个组合预计
超过一天。因此先执行明确标记为 `diagnostic_not_final` 的固定预算协议：

- 模型：OCHetGCN、GLocalKD、HGT、DeepTraLog、HRGCN、HRA-GNN；
- 每个模型三个随机种子；
- 每轮从正常训练集抽取 2000 图，最多训练 15 epochs；
- 验证和测试各按原异常比例固定分层抽取最多 4000 图；
- 所有抽样图 ID 写入 `evaluation_splits.json`。

### 9.1 TraceLog 诊断结果

| 模型 | AUROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| OCHetGCN | 0.7062 +/- 0.0603 | 0.6015 +/- 0.0551 | 0.2167 +/- 0.1662 | 0.2296 +/- 0.1674 |
| GLocalKD | 0.6795 +/- 0.0060 | 0.6318 +/- 0.0057 | 0.4388 +/- 0.0152 | 0.4089 +/- 0.0278 |
| HGT | 0.6583 +/- 0.0582 | 0.5532 +/- 0.0662 | 0.2277 +/- 0.1195 | 0.2557 +/- 0.1014 |
| DeepTraLog | 0.6472 +/- 0.0163 | 0.5416 +/- 0.0430 | 0.1228 +/- 0.1171 | 0.1210 +/- 0.1666 |
| HRGCN | 0.7381 +/- 0.0441 | 0.6961 +/- 0.0321 | **0.4795 +/- 0.0318** | **0.4658 +/- 0.0277** |
| HRA-GNN | **0.8150 +/- 0.0123** | **0.7497 +/- 0.0225** | 0.4482 +/- 0.0790 | 0.4341 +/- 0.0663 |

HRA-GNN 的排序指标 AUROC/AP 最好且方差较小，但 HRGCN 的阈值型 F1/MCC
略高。论文必须同时报告并解释两类指标，不能只保留 HRA-GNN 占优的列。

### 9.2 下一步

已配置 `direct_baselines_flowgraph_diagnostic.yaml`，使用三个随机种子和
15 epochs。FlowGraph 数据规模较小，不截断训练或评估图。

### 9.3 FlowGraph 诊断结果

| 模型 | AUROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| OCHetGCN | 0.9557 +/- 0.0561 | 0.8898 +/- 0.1494 | 0.5299 +/- 0.5007 | 0.4754 +/- 0.5257 |
| GLocalKD | 0.9561 +/- 0.0082 | 0.9738 +/- 0.0025 | 0.9727 +/- 0.0029 | 0.9526 +/- 0.0054 |
| HGT | 0.5374 +/- 0.4560 | 0.4960 +/- 0.1967 | 0.0000 +/- 0.0000 | -0.0693 +/- 0.0745 |
| DeepTraLog | **0.9785 +/- 0.0228** | **0.9856 +/- 0.0135** | **0.9796 +/- 0.0136** | **0.9644 +/- 0.0235** |
| HRGCN | 0.9333 +/- 0.1155 | 0.8672 +/- 0.2300 | 0.6650 +/- 0.5759 | 0.6438 +/- 0.6093 |
| HRA-GNN | 0.6646 +/- 0.5489 | 0.7462 +/- 0.4131 | 0.6463 +/- 0.5597 | 0.6110 +/- 0.5810 |

FlowGraph 上多个基础模型接近 1.0，说明该数据集存在容易利用的结构或规模信号。
HRA-GNN 原始乘法评分的逐 seed AUROC 为 0.9734、0.9896、0.0308。
seed 33 的 SVDD AUROC 为 0.0308，而自监督分量 AUROC 为 0.9693，说明单个
分量失效会直接拖垮乘法评分；不得把该 seed 删除或事后翻转。

## 10. 正常分位数鲁棒评分扩展

新增 `calibrated-rescore` 入口，仅使用正常训练图进行校准：

1. 分别建立 SVDD 分数和自监督异常分数的正常经验分布；
2. 把每个测试分量映射为其在正常分布中的经验分位数；
3. 使用两个分位数的最大值作为扩展异常分数；
4. 阈值仍来自正常训练扩展分数的 99% 分位数。

该方法不读取测试标签选择分数方向，也不修改已训练模型。

| 数据集 | 评分 | AUROC | AP | F1 | MCC |
|---|---|---:|---:|---:|---:|
| FlowGraph | 原论文乘法 | 0.6646 +/- 0.5489 | 0.7462 +/- 0.4131 | 0.6463 +/- 0.5597 | 0.6110 +/- 0.5810 |
| FlowGraph | 正常 ECDF-Max | **0.9689 +/- 0.0080** | **0.9726 +/- 0.0065** | **0.9710 +/- 0.0029** | **0.9495 +/- 0.0054** |
| TraceLog | 原论文乘法 | **0.8150 +/- 0.0123** | **0.7497 +/- 0.0225** | 0.4482 +/- 0.0790 | 0.4341 +/- 0.0663 |
| TraceLog | 正常 ECDF-Max | 0.7780 +/- 0.0302 | 0.7170 +/- 0.0511 | **0.4705 +/- 0.0652** | **0.4534 +/- 0.0498** |

结论：ECDF-Max 显著改善 FlowGraph 稳定性，但降低 TraceLog 的 AUROC/AP，
因此当前只能作为鲁棒评分扩展和消融实验，不能直接替换论文原乘法公式。

## 11. HDFS 与 ADFA-LD 数据集可用性

### 11.1 数据范围

| 数据集 | 图数 | 训练正常 | 验证正常 | 测试正常/异常 | 备注 |
|---|---:|---:|---:|---:|---|
| HDFS-100k | 7940 | 4576 | 1525 | 1526 / 313 | 10 万行结构化子集，不是完整 HDFS_v1 |
| ADFA-LD | 5951 | 833 | 2186 | 2186 / 746 | 完整三目录数据 |

HDFS 图节点数中位数为 13、最大 249；ADFA-LD 中位数为 343、最大 4494。

### 11.2 统计基线

| 数据集 | 模型 | AUROC | AP | TPR@1%FPR |
|---|---|---:|---:|---:|
| HDFS-100k | Isolation Forest | 0.6615 | 0.2755 | 0.0032 |
| HDFS-100k | One-Class SVM | 0.7501 | 0.5890 | 0.4920 |
| ADFA-LD | Isolation Forest | 0.4541 | 0.2192 | 0.0000 |
| ADFA-LD | One-Class SVM | 0.4796 | 0.3462 | 0.0000 |

HDFS 有一定粗统计信号，但远弱于 FlowGraph 的近完美统计分离。ADFA-LD 的
统计基线接近或低于随机，说明图规模和类型计数不足以识别攻击。

### 11.3 轻量模型三随机种子

服务器：Tesla V100 16GB；每个模型最多 10 epochs。

| 数据集 | 模型 | AUROC | AP | F1 | MCC |
|---|---|---:|---:|---:|---:|
| HDFS-100k | OCHetGCN | 0.7368 +/- 0.0124 | 0.6380 +/- 0.0101 | 0.6467 +/- 0.0051 | 0.6411 +/- 0.0111 |
| HDFS-100k | DeepTraLog-adapted | 0.7379 +/- 0.0120 | 0.6389 +/- 0.0035 | 0.6440 +/- 0.0035 | 0.6349 +/- 0.0051 |
| HDFS-100k | GLocalKD-adapted | **0.7559 +/- 0.0057** | **0.6467 +/- 0.0019** | 0.6450 +/- 0.0008 | 0.6348 +/- 0.0000 |
| ADFA-LD | OCHetGCN | **0.7938 +/- 0.0030** | **0.5092 +/- 0.0029** | **0.5154 +/- 0.0117** | **0.3973 +/- 0.0108** |
| ADFA-LD | DeepTraLog-adapted | 0.7608 +/- 0.0038 | 0.4394 +/- 0.0061 | 0.2270 +/- 0.0412 | 0.1038 +/- 0.0425 |
| ADFA-LD | GLocalKD-adapted | 0.7236 +/- 0.0111 | 0.4164 +/- 0.0134 | 0.2503 +/- 0.0229 | 0.1635 +/- 0.0155 |

### 11.4 可用性结论

- **HDFS 可用：**结果稳定、计算便宜、难度适中，下一步替换为完整 HDFS_v1；
- **ADFA-LD 可用：**图模型显著超过统计基线，能检验序列/结构建模能力；
- **ADFA-LD 有适配要求：**必须使用高基数友好的 edge-only 关系模式；
- 两者均比 FlowGraph 更适合补充论文说服力，但当前结果仍是 10-epoch
  `dataset_usability`，不能直接写成最终 SOTA 对比。
