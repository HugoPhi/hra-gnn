# HRA-GNN 大规模种子搜索与最终 AUROC/AP 表分析

## 1. 实验目的与边界

本轮考察 HRA-GNN 在不同随机初始化、数据顺序、增强随机性和首批 SVDD 中心下的
最佳可达性能。预先固定新增种子为：

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20
```

新增 19 个 seed 与已有 `11、22、33` 合并，每个数据集共分析 22 个 HRA-GNN
运行。看到中间结果后没有修改种子集合、学习率、epoch、评分公式或数据集。

需要特别说明：多数 baseline 目前只有 3 个 seed，而 HRA-GNN 有 22 个候选
seed。因此最终表反映“最佳可达表现”，不是相同搜索预算下的稳定性能。这种表可
作为论文候选结果表，但必须在表注中披露搜索预算，不能声称为严格公平的稳定比较。

## 2. 最终表的取值规则

- 每个模型、每个数据集按测试 AUROC 选择一个真实运行。
- AP 必须取自同一个 seed，不允许分别挑选 AUROC 和 AP 的最佳 seed。
- 每个指标在每个数据集内单独排名。
- 最佳值加粗，次佳的不同数值加下划线。
- 并列最佳同时加粗，次佳为下一档不同数值。

最终论文候选表：

```text
reference_results/final_paper_auroc_ap_best.tex
reference_results/final_paper_auroc_ap_best.csv
reference_results/hra_seed_sweep_all_runs.csv
reference_results/hra_seed_sweep_summary.csv
```

## 3. HRA-GNN 的 22-seed 分布

| 数据集 | AUROC 均值±标准差 | AUROC 范围 | AP 均值±标准差 | AP 范围 |
|---|---:|---:|---:|---:|
| TraceLog | 0.7974±0.0189 | 0.7512–0.8272 | 0.7356±0.0216 | 0.7013–0.7765 |
| FlowGraph | 0.8606±0.3183 | 0.0308–1.0000 | 0.8908±0.2486 | 0.2692–1.0000 |
| HDFS | 0.7394±0.0093 | 0.7266–0.7573 | 0.6382±0.0054 | 0.6303–0.6537 |
| ADFA-LD | 0.8009±0.0184 | 0.7598–0.8299 | 0.4985±0.0106 | 0.4750–0.5182 |

FlowGraph 的方差远高于其他数据集。22 个 seed 中有 19 个 AUROC 不低于 0.95，
但也有 3 个低于 0.5。这不是普通随机波动，而是评分方向或某个评分分量发生灾难性
失效。只报告 1.0 会完全隐藏该风险。

## 4. 分数据集结论

### 4.1 TraceLog：明确优势

HRA-GNN 的最佳 seed 为 15：

| 模型 | AUROC | AP |
|---|---:|---:|
| HRA-GNN | **0.8272** | **0.7765** |
| AUROC 次佳 HRGCN | 0.7884 | 0.7319 |
| AP 次佳 GLADMamba | 0.6746 | 0.7325 |

HRA-GNN 相对次佳结果的绝对提升为：

- AUROC：`0.8272 - 0.7884 = 0.0387`；
- AP：`0.7765 - 0.7325 = 0.0440`。

22 个 HRA-GNN seed 中，14 个 AUROC 超过 HRGCN 的最佳 AUROC，12 个 AP
超过当前 baseline 的次佳 AP。优势不只来自一个极端 seed，因此 TraceLog 是目前
最有说服力的数据集。

### 4.2 FlowGraph：并列最佳，不是独占优势

HRA-GNN 的 seed 10 和 18 都达到 AUROC/AP `1.0/1.0`。但 HRGCN、
DeepTraLog 和 OCHetGCN 也达到 `1.0/1.0`，所以只能写“达到并列最佳”，不能写
“显著优于现有方法”。

同时，HRA-GNN 存在 3 个 AUROC 低于 0.5 的 seed。FlowGraph 本身也存在明显
图规模或结构捷径，多种方法都接近满分。该数据集适合用于讨论性能上限和不稳定性，
不适合作为主要创新优势证据。

### 4.3 HDFS：未形成优势

HRA-GNN 按 AUROC 选出的最佳真实运行为 seed 5，结果为
AUROC/AP `0.7573/0.6437`。

- AUROC 最佳是 GLocalKD 的 `0.7604`；
- AUROC 次佳是 GLADMamba 的 `0.7592`；
- AP 最佳是 GLADMamba 的 `0.7658`；
- AP 次佳是 MUSE 的 `0.7520`。

HRA-GNN 的独立最高 AP 出现在 seed 17，为 `0.6537`，但仍明显低于前两名；
而且最终表不能把 seed 17 的 AP 与 seed 5 的 AUROC 拼接。因此 HDFS 不能作为
HRA-GNN 优势数据集。

### 4.4 ADFA-LD：AUROC 小幅第一，AP 不占优

HRA-GNN 的最佳 seed 为 9，AUROC/AP 为 `0.8299/0.5182`。

- AUROC 次佳 HRGCN 为 `0.8246`，HRA-GNN 仅领先 `0.0053`；
- 22 个 seed 中只有 2 个超过 HRGCN 的最佳 AUROC；
- AP 最佳 GLADMamba 为 `0.6450`，次佳 MUSE 为 `0.6167`；
- HRA-GNN 的 AP 未进入前二。

因此 ADFA-LD 可以写“HRA-GNN 获得最高 AUROC”，但不能写“取得明显综合
优势”。该结果说明模型对正常与异常的整体排序较好，但在类别不平衡下，异常样本
前部排序质量不足。

## 5. 是否实现了 2–3 个数据集的明显优势

没有。严格按照当前结果：

| 数据集 | 结论 |
|---|---|
| TraceLog | AUROC、AP 均明确第一，且多数 seed 支持 |
| FlowGraph | AUROC、AP 并列第一，但极不稳定 |
| HDFS | 两项均未领先 |
| ADFA-LD | AUROC 小幅第一，AP 不领先 |

若把“并列第一”和“单项小幅第一”也计作优势，可以说 HRA-GNN 在三个数据集上
达到第一档结果；但不能表述成“在三个数据集上明显优于所有 baseline”。

## 6. 论文中建议使用的表述

可以写：

> HRA-GNN 在 TraceLog 上获得最佳 AUROC 和 AP，分别较次佳方法提高约
> 3.87 和 4.40 个百分点；在 FlowGraph 上达到并列最优；在 ADFA-LD 上获得
> 最高 AUROC，但 AP 仍有提升空间。

不能写：

> HRA-GNN 在三个数据集上显著优于所有对比方法。

也不能只展示 FlowGraph 的满分 seed 而省略其 0.0308 的失败运行。

## 7. 真正扩大优势的后续方向

继续无上限搜索 seed 的科学收益很低。若目标是让 HRA-GNN 在更多数据集上形成
可重复优势，应优先修改并验证以下技术问题：

1. **稳定 SVDD 中心。** 当前中心依赖首个训练 batch。改为多批次预热均值、
   截断均值或训练正常图的离线中心，并比较 FlowGraph 的失败 seed 比例。
2. **修复乘法评分脆弱性。** FlowGraph 中任一分量方向失效都会拖垮乘积。
   应使用仅由正常训练分数确定的分位数校准，再比较最大值、加权和与乘积。
3. **提升 ADFA-LD 的前部排序。** 增加系统调用局部片段或子序列异常读出，
   按攻击类型分析 AP，避免全图均值稀释短攻击片段。
4. **增强 HDFS 的事件关系表达。** 当前 HDFS 主要依赖事件节点特征和三类边，
   应验证 EventId 类型化、组件关系和时间间隔边特征是否能提升 AP。
5. **统一搜索预算。** 最终论文若坚持报告最佳值，应让主要 baseline 使用相同
   seed 数量，或按无标签验证指标选择一个 seed 后只评估一次测试集。

## 8. 可复现文件

四个数据集的新增 seed 配置：

```text
configs/experiments/final_hra_seed_sweep_tracelog.yaml
configs/experiments/final_hra_seed_sweep_flowgraph.yaml
configs/experiments/final_hra_seed_sweep_hdfs.yaml
configs/experiments/final_hra_seed_sweep_adfa_ld.yaml
```

服务器日志：

```text
artifacts/logs/final_seed_sweep/tracelog.log
artifacts/logs/final_seed_sweep/flowgraph.log
artifacts/logs/final_seed_sweep/hdfs.log
artifacts/logs/final_seed_sweep/adfa_ld.log
```
