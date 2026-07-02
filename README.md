# HRA-GNN 论文复现工程

本仓库复现中文论文《一种基于关系偏差调制注意力的异构图异常检测算法》。
工程基于原始 HRGCN 仓库继续实现：

<https://github.com/jiaxililearn/HRGCN>

原始 HRGCN 代码保留在 `src/`，新实现位于 `hra_gnn/`。训练、评估、测试、
数据检查、实验套件和绘图均通过 `run.py` 统一调用，参数统一由 YAML 管理。

## 先读这些文档

- [复现状态与实测结果](doc/复现状态.md)
- [代码与论文实现详解](doc/代码与论文实现详解.md)
- [MathType 评分公式恢复证据](doc/评分公式恢复证据.md)
- [数据集与模型评测工作拆分](doc/数据集与模型评测工作拆分.md)

其中《代码与论文实现详解》是严格审查模型实现时的主文档，包含公式到代码的
逐项对应、张量形状、训练流程、工程优化、已知缺陷和建议审查顺序。

## 已实现内容

- 异构关系三元组：`(源节点类型, 边类型, 目标节点类型)`。
- 每类关系独立的消息变换矩阵。
- 按层、按关系维护正常原型、尺度和更新次数。
- 仅使用原始训练图更新原型的指数滑动平均。
- 标准化关系偏差和偏差调制的语义注意力。
- 关系静态融合、普通动态注意力等消融模式。
- 最大池化、均值池化和自适应门控混合读出。
- 固定中心的 DeepSVDD 单类学习目标。
- 原始图与增强图二分类自监督分支。
- 边扰动、边添加、节点类型交换和边类型交换。
- 从原始 Word 的 MathType 对象恢复出的论文联合异常评分。
- 五随机种子实验、消融实验、参数敏感性实验和效率记录。
- 预测结果、关系注意力诊断、数据集捷径诊断和独立绘图流程。
- 断连图批处理和关系聚合向量化，不改变图之间的独立性。

## 目录结构

```text
configs/                 数据集、模型和实验套件的 YAML 配置
doc/                     除 README 外的中文说明文档
hra_gnn/                 HRA-GNN 新实现
scripts/                 数据下载、重评分和复现实用脚本
src/                     原始 HRGCN 仓库代码，未覆盖
tests/                   单元测试与端到端测试
reference_results/       从论文转录的结构化结果
data/                    下载后的数据集，不纳入 Git
artifacts/results/       指标、预测、配置快照和 checkpoint
artifacts/figures/       仅由已保存结果生成的图片
run.py                   统一命令行入口
```

训练代码不会直接生成图片。绘图命令只读取 CSV 结果文件，从而保证数据结果与
绘图逻辑分离。

## 环境

项目已在以下环境验证：

- Python 3.10.17
- PyTorch 2.7（本地）
- PyTorch 2.8 + CUDA 12.8（双 RTX 4090 服务器）

本地环境已经位于 `code/.venv`：

```bash
cd code
source .venv/bin/activate
python -m pytest
```

重新创建环境：

```bash
pload 3.10.17-torch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前 Apple 设备上，MPS 执行稀疏 `index_add`/scatter 运算明显慢于 CPU，因此
`device: auto` 只会优先选择 CUDA；没有 CUDA 时使用 CPU。

## 数据集

下载并解压 TraceLog 和 FlowGraph：

```bash
.venv/bin/python scripts/download_data.py all
```

也可以分别下载：

```bash
.venv/bin/python scripts/download_data.py flowgraph
.venv/bin/python scripts/download_data.py tracelog
```

预期目录：

```text
data/ProcessedData_FlowGraph/
data/ProcessedData_TraceLog/
```

HDFS 使用 Loghub 已解析文件构图：

```bash
.venv/bin/python run.py prepare-data \
  --kind hdfs \
  --input data/raw/HDFS/HDFS.log_structured.csv \
  --labels data/raw/HDFS/anomaly_label.csv \
  --output data/ProcessedData_HDFS
```

ADFA-LD 使用官方三个轨迹目录构图：

```bash
.venv/bin/python run.py prepare-data \
  --kind adfa-ld \
  --input data/raw/ADFA-LD \
  --output data/ProcessedData_ADFA_LD
```

两者输出为可内存映射的 packed 图数据，不要求把全部图对象一次性载入内存。

当前实现使用 HRGCN 发布的图 ID 划分文件。TraceLog 有独立的训练、验证和测试
划分；FlowGraph 只有训练划分和一个评估划分。

### FlowGraph 的重要数据问题

发布数据中的 FlowGraph `edge_index.csv` 没有 `edge_type` 列。论文所称的
“26 种边类型”实际以 26 个事件编码特征列的形式出现在节点特征文件中。因此，
可执行配置使用 8 种节点类型和 1 种显式边类型。

代码没有凭空构造 26 种边标签。论文表述、发布数据与可执行表示之间的不一致
必须在修改稿中说明。

## 统一命令

### 检查数据

```bash
.venv/bin/python run.py data-info --config configs/flowgraph.yaml
.venv/bin/python run.py data-info --config configs/tracelog.yaml
.venv/bin/python run.py data-info --config configs/hdfs.yaml
.venv/bin/python run.py data-info --config configs/adfa_ld.yaml
```

### 训练

```bash
.venv/bin/python run.py train --config configs/flowgraph.yaml
.venv/bin/python run.py train --config configs/tracelog.yaml
```

可以通过命令行覆盖任意 YAML 参数：

```bash
.venv/bin/python run.py train \
  --config configs/flowgraph.yaml \
  --set training.epochs=10 \
  --set model.deviation_weight=0.5 \
  --set output.run_name=flow_lambda_05
```

### 评估 checkpoint

```bash
.venv/bin/python run.py evaluate \
  --config configs/flowgraph.yaml \
  --checkpoint artifacts/results/FlowGraph/hra_full/seed_42/checkpoints/best.pt \
  --set evaluation.collect_diagnostics=true \
  --split test
```

`test` 是同一评估入口的别名：

```bash
.venv/bin/python run.py test \
  --config configs/flowgraph.yaml \
  --checkpoint artifacts/results/FlowGraph/hra_full/seed_42/checkpoints/best.pt \
  --split test
```

### 运行五随机种子主实验

```bash
.venv/bin/python run.py experiment \
  --suite configs/experiments/full_flowgraph.yaml
.venv/bin/python run.py experiment \
  --suite configs/experiments/full_tracelog.yaml
```

### 运行基线对比

```bash
.venv/bin/python run.py experiment \
  --suite configs/experiments/baselines_flowgraph.yaml
.venv/bin/python run.py experiment \
  --suite configs/experiments/baselines_tracelog.yaml
```

### 运行论文消融和参数敏感性实验

```bash
.venv/bin/python run.py experiment \
  --suite configs/experiments/paper_flowgraph.yaml
.venv/bin/python run.py experiment \
  --suite configs/experiments/paper_tracelog.yaml
```

### 运行增强策略和原型分析

```bash
.venv/bin/python run.py experiment \
  --suite configs/experiments/augmentation_tracelog.yaml
.venv/bin/python run.py experiment \
  --suite configs/experiments/prototype_tracelog.yaml
```

实验套件将每个随机种子的结果以及均值、标准差写入：

```text
artifacts/results/suites/<实验套件名称>/
```

实验默认根据已有 `metrics.json` 断点续跑。只有明确需要覆盖已完成结果时才使用
`--force`。

## 多指标和 LaTeX 大表

每次最终测试统一输出：

- AUROC、AP；
- Precision@1%、Recall@1%、TPR@1%FPR；
- Precision、Recall、F1、MCC；
- 参数量、训练/推理时间和 CPU/GPU 峰值内存。

F1/MCC 的阈值来自正常训练分数的 99% 分位数，不使用测试标签。训练最佳
checkpoint 由无标签验证 SVDD 损失选择，AUC/AP 仅用于监控。

合并多个实验套件并生成 LaTeX：

```bash
.venv/bin/python run.py table \
  --input artifacts/results/suites/baselines_tracelog/runs.csv \
  --input artifacts/results/suites/baselines_flowgraph/runs.csv \
  --input artifacts/results/suites/baselines_hdfs/runs.csv \
  --input artifacts/results/suites/baselines_adfa_ld/runs.csv \
  --summary-csv artifacts/results/tables/all_models_summary.csv \
  --output artifacts/results/tables/all_models_all_metrics.tex
```

生成的表使用 `booktabs`、`graphicx` 和 `rotating`。

## 官方近期 Baseline

SIGNET、CVTGAD、MUSE、GLADMamba 的官方仓库和 commit 已锁定在
`configs/baselines.lock.yaml`。下载到不纳入 Git 的 `external/`：

```bash
.venv/bin/python scripts/fetch_baselines.py
```

官方协议结果只负责验证实现；公平主表将使用本项目冻结的数据划分、共同输入和
无标签 checkpoint 选择。详情见[数据集与模型评测工作拆分](doc/数据集与模型评测工作拆分.md)。

## 绘图

主结果图：

```bash
.venv/bin/python run.py plot \
  --kind comparison \
  --input artifacts/results/suites/full_main_summary.csv \
  --output artifacts/figures/paper/full_main_five_seed.png
```

论文转录结果图：

```bash
.venv/bin/python run.py plot \
  --kind comparison \
  --input reference_results/paper_main_comparison.csv \
  --output artifacts/figures/reference/main_comparison.png
```

消融和参数敏感性图：

```bash
.venv/bin/python run.py plot \
  --kind ablation \
  --input artifacts/results/suites/paper_tracelog/variant_summary.csv \
  --output artifacts/figures/paper/tracelog_ablation.png

.venv/bin/python run.py plot \
  --kind sensitivity \
  --input artifacts/results/suites/paper_tracelog/sweep_summary.csv \
  --output artifacts/figures/paper/tracelog_sensitivity.png
```

关系注意力诊断图：

```bash
.venv/bin/python run.py plot \
  --kind relations \
  --input artifacts/results/TraceLog/hra_full/seed_42/test_relations.csv \
  --output artifacts/figures/paper/tracelog_relations.png
```

## 基线实现的边界

公共实验管线中提供以下可运行模型：

- `OCHetGCN`：静态关系融合、最大池化和 DeepSVDD。
- `HRGCN`：静态关系拼接等价实现、最大池化和 SSL。
- `HGT`：类型感知 Transformer 注意力、最大池化和 DeepSVDD。
- `DeepTraLog`：拓扑 GGNN、注意力读出和 DeepSVDD。
- `GLocalKD`：固定随机教师、可训练学生和局部/全局蒸馏残差。
- `HRA-GNN`：关系偏差注意力、门控混合读出和 SSL。

其中 HGT、DeepTraLog 和 GLocalKD 是为了统一数据与评估协议编写的“论文式适配
实现”，不是对应论文作者代码的逐行复现。相关结果必须标为“适配器复现”，不能
冒充“官方实现复现”。

## FlowGraph 数据捷径诊断

```bash
.venv/bin/python run.py diagnose --config configs/flowgraph.yaml
.venv/bin/python run.py plot \
  --kind diagnostics \
  --input artifacts/results/diagnostics/FlowGraph/graph_statistics.csv \
  --output artifacts/figures/diagnostics/flowgraph_size_distribution.png
```

只使用图规模及节点/边类型计数时：

- Isolation Forest：AUC 0.968，AP 0.882。
- One-Class SVM：AUC 0.9984，AP 0.9981。

这说明 FlowGraph 几乎可以依靠简单分布特征分开，论文中的 1.0 不能直接证明
GNN 学到了关系级异常机制。

## 论文评分公式

从原始 Word 的 MathType OLE/MTEF 对象恢复出的公式为：

```text
S_dist(G) = ||g - c||_2^2
S_ssl(G) = 1 - p
S_anomaly(G) = S_dist(G) * (1 + S_ssl(G))
```

对应配置为：

```yaml
evaluation:
  score_mode: paper_product
```

每张图仍分别保存：

- `svdd_score`
- `ssl_anomaly_score`
- 联合 `score`

详细恢复依据见 [评分公式恢复证据](doc/评分公式恢复证据.md)。

## TensorBoard 训练监控

启用逐 epoch 监控：

```bash
.venv/bin/python run.py train \
  --config configs/tracelog.yaml \
  --set monitoring.enabled=true \
  --set output.run_name=monitor_tracelog
```

启动面板：

```bash
.venv/bin/tensorboard \
  --logdir artifacts/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

面板按数据集分别提供三个自定义图，每张图只有两条线：

- `SVDD Loss: train/test`：训练 batch 的平均 SVDD 损失与监控测试集平均 SVDD 距离；
- `AUC: validation/test`：监控验证集与监控测试集 AUC；
- `AP: validation/test`：监控验证集与监控测试集 AP。

训练集只包含正常图，真实 `train AUC/AP` 在数学上没有定义，所以不能伪造
`train/test AUC/AP` 曲线。TraceLog 使用现有 validation/test；FlowGraph 没有官方
验证文件，监控模块将官方评估集按标签固定拆成互斥的 validation/test 视图。该拆分只用于
诊断曲线，不改变训练数据、早停规则或最终完整测试集指标。

逐 epoch 测试曲线可能导致人工测试集调参，因此只能用于诊断，不得据此选择模型。正式
论文结果仍以早停规则选出的 `best.pt` 在完整测试集上的一次评估为准。

## 当前验证结果

- 30 项自动化测试通过。
- Ruff 静态检查通过。
- Python 编译检查通过。
- 合成数据的训练、评估、实验聚合和所有绘图入口通过。
- FlowGraph 和 TraceLog 官方数据均可加载。
- 双 RTX 4090 五随机种子结果：
  - FlowGraph：AUC `0.7204 +/- 0.4120`，AP `0.7400 +/- 0.3401`。
  - TraceLog：AUC `0.8113 +/- 0.0272`，AP `0.7293 +/- 0.0402`。

FlowGraph 五次 AUC 分别为 `0.9836、0.9837、0.6414、0.9622、0.0308`，
表现出严重的随机种子敏感性。不要只选择其中一次高分作为最终结论。

一键检查和完整实验入口：

```bash
scripts/quick_check.sh
scripts/reproduce_paper.sh
```

完整实验包含大量五随机种子任务，不是快速测试。

## 原始 HRGCN 引用

```bibtex
@inproceedings{li2023hrgcn,
  title={HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks},
  author={Li, Jiaxi and Pang, Guansong and Chen, Ling and Namazi-Rad, Mohammad-Reza},
  booktitle={10th IEEE International Conference on Data Science and Advanced Analytics},
  year={2023}
}
```
