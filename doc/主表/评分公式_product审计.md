# 评分公式 product 审计

## 1. 问题

当前默认评分为：

\[
S_{\mathrm{paper}}(G)=d_{\mathrm{SVDD}}(G)\cdot(1+a_{\mathrm{SSL}}(G)).
\]

其中：

\[
a_{\mathrm{SSL}}(G)=1-\sigma(\operatorname{SSLHead}(z_G)).
\]

用户指出论文中的联合评分可能应为直接乘法：

\[
S_{\mathrm{product}}(G)=d_{\mathrm{SVDD}}(G)\cdot a_{\mathrm{SSL}}(G).
\]

需要特别区分概率方向。训练时原始正常图的 SSL 标签为 1，增强扰动图标签为 0，
因此 \(\sigma(\operatorname{SSLHead}(z_G))\) 越大表示越像原始正常图；
异常方向应使用 \(1-\sigma(\cdot)\)，而不是 \(\sigma(\cdot)\)。

## 2. 零训练重评分

使用已有 `test_predictions.csv` 中的 `svdd_score`、`ssl_anomaly_score` 和
`ssl_probability`，不重新训练，只重新计算 AUROC/AP。结果保存在：

```text
reference_results/scoring_formula_audit/offline_rescore_runs.csv
reference_results/scoring_formula_audit/offline_rescore_summary.csv
```

可复现命令：

```bash
.venv/bin/python scripts/offline_rescore_formula.py \
  --input 'artifacts/results/TraceLog/full_tracelog/HRA-GNN/seed_*/test_predictions.csv' \
  --input 'artifacts/results/FlowGraph/full_flowgraph/HRA-GNN/seed_*/test_predictions.csv' \
  --output-dir reference_results/scoring_formula_audit
```

## 3. 离线结果

| 数据集 | 评分 | AUROC | AP |
|---|---|---:|---:|
| TraceLog | SVDD | 0.8110±0.0269 | 0.7296±0.0385 |
| TraceLog | SSL anomaly | 0.5311±0.2055 | 0.4597±0.1578 |
| TraceLog | SVDD × SSL anomaly | 0.8109±0.0284 | 0.7290±0.0426 |
| TraceLog | SVDD × (1 + SSL anomaly) | 0.8113±0.0272 | 0.7293±0.0402 |
| TraceLog | 错误方向：SVDD × SSL probability | 0.8075±0.0293 | 0.7265±0.0388 |
| FlowGraph | SVDD | 0.7187±0.4124 | 0.7391±0.3410 |
| FlowGraph | SSL anomaly | 0.7777±0.4297 | 0.8364±0.3173 |
| FlowGraph | SVDD × SSL anomaly | 0.7233±0.4112 | 0.7425±0.3377 |
| FlowGraph | SVDD × (1 + SSL anomaly) | 0.7204±0.4120 | 0.7400±0.3401 |
| FlowGraph | 错误方向：SVDD × SSL probability | 0.7173±0.4128 | 0.7382±0.3418 |

离线结论：

1. 直接乘法必须使用 `ssl_anomaly_score = 1 - sigmoid(logit)`。
2. 使用 `ssl_probability = sigmoid(logit)` 的乘法方向错误，在 TraceLog 上略差。
3. 在已有模型输出上，`product` 与 `paper_product` 差异很小；FlowGraph 上
   `product` 略高，TraceLog 上 `paper_product` 略高。
4. 离线重评分不能替代重训，因为评分函数也影响验证 AUROC/AP 和 checkpoint 选择。

## 4. 重训验证

离线重评分只能说明同一组模型输出下不同评分公式的差异，不能说明训练过程中
checkpoint 选择受到评分公式影响后的最终结果。因此额外做了一组最小重训实验：

- 数据集：TraceLog
- 配置文件：`configs/tuning/tracelog_default_product_final_test.yaml`
- 训练设置：沿用旧默认配置，只将 `evaluation.score_mode` 改为 `product`
- 种子：11、22、33、44、55
- 结果位置：
  `reference_results/scoring_formula_audit/tracelog_default_product_final_test/`

与旧默认评分 `paper_product = SVDD × (1 + SSL anomaly)` 的对比如下：

| 评分口径 | AUROC | AP | 平均训练时间 |
|---|---:|---:|---:|
| 旧默认：SVDD × (1 + SSL anomaly) | 0.8172±0.0162 | 0.7608±0.0185 | 92.55s |
| 新验证：SVDD × SSL anomaly | 0.8213±0.0202 | 0.7618±0.0177 | 95.64s |

差值：

- AUROC：`+0.0041`
- AP：`+0.0009`
- 平均训练时间：`+3.09s`

随后在 FlowGraph 上做同样的最小重训验证：

- 数据集：FlowGraph
- 配置文件：`configs/tuning/flowgraph_product_final_test.yaml`
- 训练设置：沿用 `configs/flowgraph.yaml`，只将 `evaluation.score_mode` 改为 `product`
- 种子：11、22、33、44、55
- 结果位置：
  `reference_results/scoring_formula_audit/flowgraph_product_final_test/`

FlowGraph 的重训结果如下：

| 种子 | AUROC | AP | 说明 |
|---:|---:|---:|---|
| 11 | 0.9798 | 0.9830 | 正常高分 |
| 22 | 0.9870 | 0.9876 | 正常高分 |
| 33 | 0.0308 | 0.2692 | 几乎反向排序 |
| 44 | 0.9985 | 0.9981 | 正常高分，当前最好 |
| 55 | 0.0323 | 0.2694 | 几乎反向排序 |
| 均值 | 0.6057±0.4688 | 0.7015±0.3529 | 方差极大 |

因此，FlowGraph 上 `product` 公式可以在好种子上取得接近满分结果，
但不能解决 33、55 等种子的反向排序问题。这个现象说明 FlowGraph 的主要风险是
训练稳定性、checkpoint 选择或异常方向校准，而不是 `1 + a_SSL` 与 `a_SSL`
之间的简单公式差异。

完整对照表保存为：

```text
reference_results/scoring_formula_audit/scoring_formula_comparison.csv
```

## 5. 结论

1. 从公式语义看，若论文希望表达“SVDD 异常程度和 SSL 异常程度共同高时才给高分”，
   更直接的写法应为：
   \[
   S(G)=d_{\mathrm{SVDD}}(G)\cdot(1-\sigma(\operatorname{SSLHead}(z_G))).
   \]
2. 概率方向必须使用 `1 - sigmoid(logit)`。因为 SSL 训练中原图标签为 1，
   `sigmoid(logit)` 表示“像原始正常图”，不是异常概率。
3. 小范围重训显示，`product` 在 TraceLog 默认配置上略优于 `paper_product`，
   但提升幅度很小，不能把它作为主表显著收益的主要来源。
4. FlowGraph 上 `product` 的最好种子可达到 AUROC 0.9985、AP 0.9981，
   但五种子均值不稳定，不建议用它声称“公式修正带来稳定提升”。
5. 后续如果要完全切换论文公式，应重跑最终主表配置；如果时间紧，可以在论文中将
   该处作为评分公式修正与消融说明，而主表仍以已经完成的最优配置为准。
