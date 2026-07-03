# 主表实验文档索引

本目录集中保存论文主表涉及的数据、模型、评分扩展、服务器记录和写作稿。

## 建议阅读顺序

1. [论文实验部分写作稿](论文实验部分写作稿.md)
2. [数据集说明与派生图构建过程](数据集说明与派生图构建过程.md)
3. [主表第一次实验](主表第一次实验.md)
4. [HRA-GNN 大规模种子搜索与最终 AUROC/AP 表分析](HRA-GNN大规模种子搜索与最终AUROC_AP表分析.md)
5. [FlowGraph ECDF-Max 与 ADFA-LD 序列增强评分详解](FlowGraph_ECDF-Max与ADFA-LD序列增强评分详解.md)
6. [ADFA-LD AP 改进与固定协议复评](ADFA-LD_AP改进与固定协议复评.md)
7. [近期 Baseline 服务器验收记录](近期Baseline服务器验收记录.md)
8. [数据集与模型评测工作拆分](数据集与模型评测工作拆分.md)

## 文件用途

| 文件 | 用途 | 是否可直接入稿 |
|---|---|---|
| `论文实验部分写作稿.md` | 数据集、划分、模型参数、训练参数、设备和主结果写作 | 可作为修改稿底稿，需按文中审计项复核 |
| `数据集说明与派生图构建过程.md` | 四个数据集的字段、构图和标签追踪 | 方法和数据集小节的证据 |
| `主表第一次实验.md` | 63 个服务器运行及第一轮受控预算 | 实验台账，不宜整段入稿 |
| `HRA-GNN大规模种子搜索与最终AUROC_AP表分析.md` | 22 seed 最佳运行和稳定性 | 结果分析及限制 |
| `FlowGraph_ECDF-Max与ADFA-LD序列增强评分详解.md` | 两项评分扩展的技术原因和实现 | 方法扩展、消融和讨论 |
| `ADFA-LD_AP改进与固定协议复评.md` | ADFA-LD 固定 1000 图协议 | ADFA-LD 主表和消融证据 |
| `近期Baseline服务器验收记录.md` | 官方实现 commit、patch 和 V100 环境 | 复现附录证据 |
| `数据集与模型评测工作拆分.md` | 工作完成状态与剩余风险 | 内部交接材料 |

## 结构化主结果

```text
reference_results/final_paper_auroc_ap_best.tex
reference_results/final_paper_auroc_ap_best.csv
reference_results/adfa_ld_fixed1000_best.csv
reference_results/adfa_ld_hybrid_ablation_seed9.csv
reference_results/hra_seed_sweep_all_runs.csv
reference_results/hra_seed_sweep_summary.csv
```

当前最终候选表只保留 AUROC 和 AP，并在每个数据集内对最佳和次佳结果分别使用
粗体和下划线。

## 投稿前公平性检查

当前候选主表已经足以判断方法方向和组织论文结果，但还不是完全相同预算的严格
公平表：

- HRA-GNN 汇总 22 个 seed，多数 baseline 汇总 3 个 seed；
- 近期方法和直接基线在 TraceLog、FlowGraph、HDFS 上使用过不同采样上限；
- 早期双 RTX 4090 性能记录与最终 V100 主表记录并存，效率表只能使用同一轮
  V100 协议下的记录；
- ADFA-LD 已统一为固定 1000 图，但其他三个数据集尚未全部固定到同一图 ID；
- 当前 HDFS 是 HDFS-100k 派生子集，论文中不能写成完整 HDFS_v1；
- ADFA-LD 当前第三类边缺少独立业务语义，正式实验建议移除后重跑。

这些问题不要求删除现有结果。正确做法是保留当前表作为“最佳可达性能”，并在
投稿前增加一张统一 seed、统一图 ID、统一设备的公平复核表。
