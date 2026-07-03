# FlowGraph ECDF-Max 与 ADFA-LD 序列增强评分详解

> 本文归档于 `doc/主表/`，详细解释两项主表评分扩展的诊断和实现过程。

## 1. 文档目的

本文单独解释两项评分改进：

1. FlowGraph 上使用正常训练集 ECDF-Max，解决原论文乘法评分对随机种子极度
   敏感的问题；
2. ADFA-LD 上使用 HRA-GNN SVDD、系统调用词频近邻和三阶 Markov 的正常
   ECDF 融合，提高 AUROC 和 AP。

两项改进都使用正常训练图建立参考分布，但它们解决的不是同一个问题：

| 数据集 | 原问题 | 解决逻辑 | 最终融合 |
|---|---|---|---|
| FlowGraph | SVDD 或 SSL 中一个分量可能失效，乘法被错误分量控制 | 统一量纲后，只要一个分量认为异常就保留告警 | ECDF 后取最大值 |
| ADFA-LD | 图级 SVDD 缺少系统调用频率和局部顺序证据 | 将图关系、调用组成和调用顺序三类互补证据累积 | ECDF 后加权求和 |

这两种方法都不是重新训练一个更大的 GNN。它们加载已经训练好的 checkpoint，
只使用正常训练集拟合校准器，再对测试图重评分。

## 2. 共同基础：正常经验分布校准

### 2.1 为什么不同分数不能直接相加或比较

HRA-GNN 会产生多种异常信号。例如：

- SVDD 距离可能在 `0.001` 到 `0.1` 之间；
- 自监督异常分数位于 `0` 到 `1`；
- 余弦近邻距离位于 `0` 到 `2`；
- Markov 平均负对数似然没有固定上界。

原始数值的尺度不同。直接相加时，大尺度分量会仅因数值范围更大而占主导，并不
代表它包含更多异常信息。

### 2.2 本项目中的 ECDF 定义

设正常训练集某个分量的分数为：

\[
\mathcal{R}=\{r_1,r_2,\ldots,r_n\}
\]

对任意待评估分数 \(x\)，定义正常经验分位数：

\[
\widehat F_{\mathcal{R}}(x)
=\frac{\sum_{i=1}^{n}\mathbf{1}(r_i\le x)}{n+1}
\]

代码使用排序数组和 `numpy.searchsorted(..., side="right")` 实现。分母采用
`n+1`，因此即使分数超过全部正常参考值，结果也不会恰好等于 1。

直观解释：

- 接近 `0.5`：处于正常训练分数的中间位置；
- 接近 `0.9`：高于约 90% 的正常训练图，比较可疑；
- 接近 `0`：低于几乎全部正常训练图。

ECDF 在这里是秩变换，不是经过概率校准的“异常概率”。它的作用是把不同分量
转换到相同的正常参考坐标系。

实现位置：

```text
hra_gnn/rescoring.py::empirical_normal_percentile
```

### 2.3 为什么只能使用正常训练集

本项目是单类异常检测。若利用测试标签决定：

- 是否翻转某个分数；
- 选择哪个融合权重；
- 选择测试集上最好的阈值；

就会发生测试泄漏。

因此两项改进都遵守：

```text
拟合参考分布：只使用正常训练图
计算测试分数：不读取测试标签
确定分类阈值：正常训练融合分数的 99% 分位数
AUROC/AP：标签只在最终评估时使用
```

不过，方法是在观察数据集实验现象后开发的，仍属于探索性改进。投稿前应冻结公式
和权重，再通过独立划分或新数据集做确认性验证。

## 3. FlowGraph：问题是如何发现的

### 3.1 最初看到的现象

在相同配置、15 个 epoch 和 seed `11、22、33` 下，HRA-GNN 原论文乘法评分的
FlowGraph AUROC 为：

```text
seed 11    0.9734
seed 22    0.9896
seed 33    0.0308
```

三次均值和标准差为：

| 评分方式 | AUROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| 原论文乘法 | 0.6646±0.5489 | 0.7462±0.4131 | 0.6463±0.5597 | 0.6110±0.5810 |

这不是普通的小幅随机波动。seed 33 的 AUROC 接近 0，说明该 seed 的排序方向
几乎完全错误。继续只报告 seed 11 或 seed 22 会掩盖模型的不稳定性。

扩大到 22 个 HRA-GNN seed 后仍能看到同一问题：

```text
AUROC 范围：0.0308 到 1.0000
AP 范围：   0.2692 到 1.0000
```

### 3.2 先把最终分数拆开

原论文评分在代码中为：

\[
S_{\mathrm{paper}}(G)
=d_{\mathrm{SVDD}}(G)\,[1+a_{\mathrm{SSL}}(G)]
\]

其中：

\[
d_{\mathrm{SVDD}}(G)
=\operatorname{mean}\left[(z_G-c)^2\right]
\]

\[
a_{\mathrm{SSL}}(G)
=1-\operatorname{sigmoid}(\operatorname{SSLHead}(z_G))
\]

训练时，自监督头把原始正常图标为 1，把人工增强图标为 0。因此测试图越像人工
增强图，\(a_{\mathrm{SSL}}\) 越高。

由于：

\[
a_{\mathrm{SSL}}\in[0,1],
\qquad 1+a_{\mathrm{SSL}}\in[1,2]
\]

SSL 最多把 SVDD 距离放大两倍。最终排序主要由 SVDD 决定。

### 3.3 seed 33 给出的关键证据

对失败 seed 分别计算两个分量的 AUROC：

```text
SVDD AUROC：0.0308
SSL  AUROC：0.9693
最终乘法：  0.0308
```

这说明：

1. 编码器并非完全没有异常信息，因为 SSL 分量可以正确排序；
2. 失败主要来自 SVDD 中心或距离方向；
3. 原乘法不是两个分量平等融合，而是“SVDD 主导，SSL 只能有限调制”；
4. 当 SVDD 失效时，正确的 SSL 分量无法救回最终结果。

FlowGraph 学习率为 `0.01`，SVDD 中心又依赖初始化阶段的正常图表示。不同随机
初始化、首批图和训练顺序可能把表示空间及固定中心带到不同状态。FlowGraph 没有
官方 validation 划分，只能用训练损失选择 checkpoint，也很难提前识别“训练损失
正常但测试排序反向”的运行。

### 3.4 为什么不能采用几个看似简单的处理

不能删除 seed 33。它是预先选择的合法随机种子，删除会形成选择性报告。

不能根据测试 AUROC 翻转 SVDD。先观察标签再使用 `-SVDD` 会直接泄漏测试标签。

不能只改成更大的 SSL 权重。两个原始分数的量纲不同，而且不同 seed 中哪个分量
失效并不固定。

不能只报告最好 seed。最好 seed 可以说明模型的最佳可达性能，不能证明稳定性。

## 4. FlowGraph：ECDF-Max 如何解决

### 4.1 第一步：分别建立正常参考分布

使用训练划分中的正常图，分别计算：

\[
\mathcal{R}_d
=\{d_{\mathrm{SVDD}}(G_i)\}_{G_i\in\mathcal{D}_{train}}
\]

\[
\mathcal{R}_a
=\{a_{\mathrm{SSL}}(G_i)\}_{G_i\in\mathcal{D}_{train}}
\]

测试图 \(G\) 的两个分量被转换为：

\[
p_d(G)=\widehat F_{\mathcal{R}_d}(d_{\mathrm{SVDD}}(G))
\]

\[
p_a(G)=\widehat F_{\mathcal{R}_a}(a_{\mathrm{SSL}}(G))
\]

转换后，两者都表示“该图在多大程度上超过正常训练图”。

### 4.2 第二步：取最大值

最终分数为：

\[
S_{\mathrm{ECDF-Max}}(G)=\max[p_d(G),p_a(G)]
\]

它相当于 OR 规则：

```text
SVDD 明显异常  -> 分数高
SSL 明显异常   -> 分数高
两者都正常     -> 分数才低
```

为什么是最大值而不是乘法：

- 乘法要求两个分量方向都可靠，一个分量很低就会压低整体；
- 最大值允许一个可靠分量独立发出异常信号；
- ECDF 已消除量纲差异，最大值不会天然偏向数值范围更大的分量；
- 它不需要使用测试标签判断当前 seed 应信任 SVDD 还是 SSL。

ECDF-Max 并没有把 seed 33 的 SVDD 方向翻转。失效 SVDD 对正常图和异常图都
被映射到正常参考秩，而正确的 SSL 分量仍能让异常图获得较高最大值。因此它是
“允许可靠分量接管”，不是“根据答案修正错误方向”。

### 4.3 第三步：阈值仍由正常训练集确定

先对正常训练图本身计算 ECDF-Max 分数，再取其 99% 分位数：

\[
\tau=Q_{0.99}
\left(
\{S_{\mathrm{ECDF-Max}}(G_i)\}_{G_i\in\mathcal{D}_{train}}
\right)
\]

测试时：

```text
score > tau  -> 异常
score <= tau -> 正常
```

AUROC 和 AP 不依赖这个阈值；F1 和 MCC 使用该阈值。

### 4.4 实际结果

| 评分方式 | AUROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| 原论文乘法 | 0.6646±0.5489 | 0.7462±0.4131 | 0.6463±0.5597 | 0.6110±0.5810 |
| 正常 ECDF-Max | **0.9689±0.0080** | **0.9726±0.0065** | **0.9710±0.0029** | **0.9495±0.0054** |

最重要的变化不是单次最高值，而是标准差：

```text
AUROC 标准差：0.5489 -> 0.0080
AP 标准差：   0.4131 -> 0.0065
```

这说明 ECDF-Max 确实解决了“某个分量偶然失效导致整次运行崩溃”的问题。

### 4.5 为什么它不能直接替换所有数据集的评分

在 TraceLog 上：

| 评分方式 | AUROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| 原论文乘法 | **0.8150±0.0123** | **0.7497±0.0225** | 0.4482±0.0790 | 0.4341±0.0663 |
| 正常 ECDF-Max | 0.7780±0.0302 | 0.7170±0.0511 | **0.4705±0.0652** | **0.4534±0.0498** |

TraceLog 的排序指标反而下降。可能原因是两个分量在 TraceLog 上没有 FlowGraph
那样明确的“一个崩溃、另一个正确”的互补关系。最大值会放大任意一个分量的正常
尾部离群点，从而损伤精细排序。

所以 ECDF-Max 的正确定位是：

```text
FlowGraph 稳定性扩展和消融方法
```

而不是：

```text
对所有数据集无条件替换论文评分
```

## 5. FlowGraph 的实现和复现

实现文件：

```text
hra_gnn/rescoring.py
```

关键函数：

| 函数 | 作用 |
|---|---|
| `empirical_normal_percentile` | 把任意分数映射为正常训练经验分位数 |
| `calibrated_max_scores` | 对 SVDD、SSL 分别做 ECDF，再取最大值 |
| `rescore_calibrated_max` | 加载 checkpoint、计算参考分数、评估并保存结果 |

运行命令：

```bash
.venv/bin/python run.py calibrated-rescore \
  --config artifacts/results/FlowGraph/direct_baselines_flowgraph_diagnostic/HRA-GNN/seed_11/config.yaml \
  --checkpoint artifacts/results/FlowGraph/direct_baselines_flowgraph_diagnostic/HRA-GNN/seed_11/checkpoints/best.pt
```

每个运行目录输出：

```text
calibrated_max_predictions.csv
calibrated_max_metrics.json
```

预测文件保留 `graph_id、label、svdd_score、ssl_anomaly_score` 和
`calibrated_max_score`，便于检查最终分数由哪个分量触发。

## 6. ADFA-LD：最初的 AP 问题其实包含两层

### 6.1 第一层：原比较协议不一致

最初主表中：

```text
HRA-GNN AP：   0.5182
GLADMamba AP：0.6450
```

看起来 HRA-GNN 明显落后，但两者并非在相同测试集上评估：

| 方法 | 测试图数 | 正常 | 异常 | 异常比例 |
|---|---:|---:|---:|---:|
| HRA-GNN 原结果 | 2932 | 2186 | 746 | 25.4% |
| GLADMamba 现有结果 | 1000 | 500 | 500 | 50.0% |

AP 对正类比例敏感。异常比例从 25.4% 变为 50% 时，即使排序能力相同，随机基准
AP 也会从 0.254 变为 0.5。因此 `0.5182` 和 `0.6450` 不能直接比较。

本次固定：

```text
configs/splits/adfa_ld_fixed_test_1000.txt
```

所有模型都只在同一组 500 正常、500 异常图上重算指标。该子集用于统一评估，
不参与训练和评分器拟合。

### 6.2 协议统一后还剩下什么问题

同一 ADFA-LD-1000 上：

| 方法 | AUROC | AP |
|---|---:|---:|
| 原始 HRA-GNN SVDD | 0.8360 | 0.7708 |
| HRGCN | 0.8331 | 0.7789 |

可见原先“远低于 GLADMamba”的主要原因是评测协议错误。但 HRA-GNN 的 AP 仍略
低于 HRGCN，说明异常样本在排序列表前部还不够集中。

## 7. ADFA-LD：为什么图级 SVDD 不够

### 7.1 ADFA-LD 的原始信息是什么

每个 ADFA-LD 文本文件是一条 Linux 系统调用序列：

```text
5 3 3 6 42 120 ...
```

预处理时：

```text
一条轨迹       -> 一张图
一次 syscall   -> 一个节点
syscall 编号   -> node_type
相邻调用       -> 时间顺序边
同类重复调用   -> 重复依赖边
```

节点按照原始系统调用出现顺序创建，因此图对象中的：

```python
graph.node_type.tolist()
```

仍是该轨迹经过训练词表映射后的 syscall 序列。系统调用词表只由 833 条正常训练
轨迹建立，共 151 种节点类型，含 unknown 类型 0。

### 7.2 图级读出的信息压缩

HRA-GNN 的关系层能够利用相邻和重复边，但最终仍将整张图压缩为一个向量：

\[
z_G=\operatorname{Gate}
\left(
\operatorname{MaxPool}(\{h_v\}),
\operatorname{MeanPool}(\{h_v\})
\right)
\]

然后仅使用：

\[
d_{\mathrm{SVDD}}(G)
=\operatorname{mean}[(z_G-c)^2]
\]

评分。

如果一条几百到几千步的轨迹中只有一个短攻击片段，mean pooling 会稀释它；
max pooling 能保留强响应，却不直接记录“某几个 syscall 以什么局部顺序共同
出现”。这会影响 AP，因为 AP 特别关心异常样本能否被排到列表最前部。

因此 ADFA-LD 的改进目标不是再增加一层通用 GNN，而是补回两类领域信息：

1. 一条轨迹由哪些 syscall 组成，频率是否偏离正常；
2. syscall 的局部转移顺序是否符合正常模式。

## 8. ADFA-LD 的三个评分分量

### 8.1 分量一：HRA-GNN SVDD

\[
s_{\mathrm{SVDD}}(G)
=\operatorname{mean}[(z_G-c)^2]
\]

它保留 HRA-GNN 的核心信息：

- 节点类型投影；
- 关系专属消息变换；
- 正常关系原型和关系偏差调制注意力；
- 自适应 max/mean 混合读出。

它主要回答：

```text
整张系统行为图是否偏离正常图表示空间？
```

### 8.2 分量二：syscall unigram TF-IDF 最近邻

将每条轨迹转换为 syscall token 文本，例如：

```text
"5 3 3 6 42 120"
```

使用正常训练轨迹拟合 unigram TF-IDF：

```python
TfidfVectorizer(
    tokenizer=str.split,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)
```

关键含义：

- `unigram` 只统计 syscall 种类和频率，不建模顺序；
- `min_df=2` 去掉只在一条正常训练轨迹中出现的偶然 token；
- `sublinear_tf=True` 使用对数词频，避免超长轨迹中的高频调用完全支配向量；
- IDF 降低几乎每条正常轨迹都出现的 syscall 权重。

对测试轨迹 \(G\)，分数为其 TF-IDF 向量到最近正常训练轨迹的余弦距离：

\[
s_{\mathrm{unigram}}(G)
=\min_{N\in\mathcal{D}_{train}}
\left[1-\cos(v_G,v_N)\right]
\]

训练图计算自身参考分数时使用两个近邻，取第二个距离，避免“自己是自己的最近邻”
导致全部正常参考分数为 0；测试图使用一个最近邻。

它主要回答：

```text
这条轨迹的 syscall 组成和频率是否像任何一条正常轨迹？
```

### 8.3 分量三：三阶 syscall Markov 平均负对数似然

三阶 Markov 使用前两个 syscall 预测当前 syscall。对序列
\((t_1,\ldots,t_L)\)，使用正常训练轨迹统计三元组和二元上下文计数。

带 Laplace 平滑的条件概率为：

\[
P(t_i\mid t_{i-2},t_{i-1})
=\frac{C(t_{i-2},t_{i-1},t_i)+1}
{C(t_{i-2},t_{i-1})+V}
\]

其中 \(V\) 是 syscall 词表大小。序列开头补两个特殊前缀 `-1`。图分数是平均
负对数似然：

\[
s_{\mathrm{Markov}}(G)
=-\frac{1}{L}\sum_{i=1}^{L}
\log P(t_i\mid t_{i-2},t_{i-1})
\]

平均而不是求和，是为了减少序列长度本身对分数的直接控制。

它主要回答：

```text
即使 syscall 种类常见，它们的局部出现顺序是否异常？
```

## 9. ADFA-LD：为什么使用 ECDF 加权和

### 9.1 先把三个分量变成正常分位数

\[
p_s(G)=\widehat F_{\mathrm{SVDD}}(s_{\mathrm{SVDD}}(G))
\]

\[
p_u(G)=\widehat F_{\mathrm{unigram}}(s_{\mathrm{unigram}}(G))
\]

\[
p_m(G)=\widehat F_{\mathrm{Markov}}(s_{\mathrm{Markov}}(G))
\]

三个 ECDF 都只用正常训练图拟合。

### 9.2 最终公式

\[
S_{\mathrm{ADFA}}(G)
=p_s(G)+0.5p_u(G)+0.25p_m(G)
\]

权重体现以下优先级：

```text
图级 HRA-GNN 关系表示     1.00  主分量
syscall 组成和频率        0.50  中等补充
局部 Markov 顺序          0.25  较弱补充
```

为什么这里使用加权和，而 FlowGraph 使用最大值：

- FlowGraph 已有两个分量，问题是某一分量在个别 seed 灾难性失效，需要 OR
  式兜底；
- ADFA-LD 的三个分量描述不同层次的信息，单个证据可能不强，但共同出现时应
  累积；
- SVDD 仍应代表 HRA-GNN 主体，因此权重最高；
- unigram 和 Markov 不能完全接管分数，避免最终方法退化成纯序列统计模型。

## 10. ADFA-LD 的分析与实验过程

### 10.1 先尝试模型内部改动

实际验证过：

- hybrid readout 改成 max；
- hybrid readout 改成 mean；
- 关闭 SSL；
- 降低 SSL 权重；
- 直接把关系偏差的 max、mean 或 top-k 作为图异常分数。

这些方案均未超过原始 hybrid。直接关系偏差评分的 AUROC 约为
`0.23--0.35`，说明关系偏差更适合作为消息注意力调制量，不能直接当作全图异常
分数。

### 10.2 再尝试嵌入空间传统检测器

对图嵌入尝试：

- Mahalanobis 距离；
- kNN；
- One-Class SVM；
- Isolation Forest。

这些方法没有解决短系统调用片段被图级表示压缩的问题，因此没有超过最终序列
增强评分。

### 10.3 最后回到数据生成机制

ADFA-LD 的原始对象不是一般静态图，而是系统调用序列。图表示已经利用结构关系，
但不能保证保留全部 token 频率和局部顺序。于是最终保留 HRA-GNN SVDD，同时
显式加入 unigram 和 Markov 两类序列证据。

这一步的关键不是“加入更多模型一定更好”，而是：

```text
观察失败指标
-> 检查评测协议
-> 分析原始数据语义
-> 找到图级读出丢失的信息
-> 用正常训练集可拟合的最小模块补回该信息
-> 做独立分量消融
```

## 11. ADFA-LD 的结果和消融

### 11.1 分量消融

同一个 seed 9 checkpoint、同一个 ADFA-LD-1000 固定测试集：

| 评分方式 | AUROC | AP |
|---|---:|---:|
| HRA-GNN SVDD | 0.8360 | 0.7708 |
| HRA-GNN + unigram | 0.8541 | 0.8367 |
| HRA-GNN + Markov | 0.8557 | 0.8261 |
| HRA-GNN + unigram + Markov | **0.8584** | **0.8475** |

可以得到三个结论：

1. unigram 单独加入时，AP 从 `0.7708` 提升到 `0.8367`，说明 syscall 组成
   对异常前部排序很重要；
2. Markov 单独加入时，两项指标也都提高，说明调用顺序包含词频之外的信息；
3. 完整组合 AP 最高，说明词频和顺序证据并不完全重复。

### 11.2 与同协议 baseline 比较

| 模型 | AUROC | AP |
|---|---:|---:|
| HRA-GNN 序列增强评分 | **0.8584** | **0.8475** |
| HRGCN | 0.8331 | 0.7789 |
| HGT | 0.8253 | 0.7679 |
| OCHetGCN | 0.8081 | 0.7651 |
| GLADMamba | 0.7146 | 0.6450 |

相对次佳 HRGCN：

```text
AUROC 相对提升约 3.0%
AP 相对提升约 8.8%
```

两项都是最佳，但不能写成“两项都超过第二名 5%”。

### 11.3 完整测试集结果

同一评分器在完整 2932 图测试集上得到：

```text
AUROC 0.8569
AP    0.6595
```

完整集 AP 低于平衡子集是预期现象，因为异常比例从 50% 降为 25.4%。该结果说明
提升不只存在于固定子集，但其他方法没有同一完整测试集输出，因此完整集结果不进入
横向主表。

### 11.4 22 个 seed 的稳定性

固定评分结构和权重后，22 个 HRA-GNN seed 的固定子集结果为：

| 指标 | 均值±标准差 | 最佳值 |
|---|---:|---:|
| AUROC | 0.8350±0.0117 | 0.8584 |
| AP | 0.8315±0.0072 | 0.8475 |

其中：

```text
22/22 个 seed 的 AP 超过 HRGCN
11/22 个 seed 的 AUROC 超过 HRGCN
```

这说明 AP 提升较稳定，但最佳 AUROC 仍受到 seed 搜索预算影响。

## 12. ADFA-LD 的实现和复现

实现文件：

```text
hra_gnn/adfa_scoring.py
```

关键函数：

| 函数 | 作用 |
|---|---|
| `fit_markov_counts` | 从正常训练序列统计 n-gram 和上下文 |
| `markov_nll_scores` | 计算每条轨迹的平均 Markov 负对数似然 |
| `rescore_adfa_hybrid` | 加载 checkpoint，拟合三个正常参考分布并融合 |

统一入口：

```bash
.venv/bin/python run.py adfa-hybrid-rescore \
  --config artifacts/results/ADFA-LD/final_hra_seed_sweep_adfa_ld/HRA-GNN/seed_9/config.yaml \
  --checkpoint artifacts/results/ADFA-LD/final_hra_seed_sweep_adfa_ld/HRA-GNN/seed_9/checkpoints/best.pt \
  --fixed-test-ids configs/splits/adfa_ld_fixed_test_1000.txt
```

默认参数：

```text
unigram_weight = 0.5
markov_weight  = 0.25
markov_order   = 3
```

输出：

```text
adfa_hybrid_predictions.csv
adfa_hybrid_metrics.json
```

预测文件为每张图保留：

```text
graph_id
label
svdd_percentile
unigram_knn_percentile
markov_nll_percentile
hybrid_score
selected_for_fixed_test
```

因此可以逐图检查某个异常是由图表示、调用频率还是局部顺序触发。

## 13. 两项改进的代码数据流

### 13.1 FlowGraph

```text
正常训练图
  -> 已训练 HRA-GNN
  -> normal_svdd, normal_ssl
  -> 分别排序形成两个 ECDF

测试图
  -> 同一 HRA-GNN checkpoint
  -> test_svdd, test_ssl
  -> 正常 ECDF 映射
  -> max(svdd_percentile, ssl_percentile)
  -> AUROC/AP 或正常 99% 阈值分类
```

### 13.2 ADFA-LD

```text
正常训练图
  -> HRA-GNN SVDD 参考分数
  -> syscall unigram TF-IDF 正常近邻参考分数
  -> syscall trigram Markov 正常 NLL 参考分数
  -> 三个正常 ECDF

测试图
  -> HRA-GNN SVDD
  -> unigram 最近正常距离
  -> Markov 平均 NLL
  -> 三个正常 ECDF 映射
  -> 1.0 * SVDD + 0.5 * unigram + 0.25 * Markov
  -> AUROC/AP 或正常 99% 阈值分类
```

## 14. 必须明确的科研边界

### 14.1 FlowGraph

- ECDF-Max 显著提高三 seed 稳定性，但当前仍是评分扩展实验；
- 它在 TraceLog 上降低 AUROC/AP，不能宣称为通用最优融合；
- FlowGraph 本身存在图规模和类型计数捷径，简单 OCSVM 可达到约
  `0.9984/0.9981`，接近满分不能单独证明 GNN 创新有效；
- FlowGraph 发布文件实际只有一种显式边类型，与原论文描述的 26 种边类型不一致。

### 14.2 ADFA-LD

- 固定 1000 图协议解决了 AP 不可比问题，但属于平衡评测子集；
- 序列增强评分是看到原始结果后开发的探索性扩展，应在独立划分上复验；
- HRA-GNN 搜索了 22 个 seed，多数 baseline 只有 3 个，最佳值搜索预算不相同；
- 最终论文若采用该结果，方法部分、消融实验和表注必须同步增加序列评分，不能只
  替换主表数字；
- 当前 ADFA-LD 派生图的第三类边与相邻边基本重复，最终正式实验仍建议移除该
  冗余边后完整重跑。

## 15. 论文中应该如何描述

FlowGraph 可以写：

> 对逐分量结果的分析表明，原乘法评分在 SVDD 分量失效时无法利用仍然有效的
> 自监督信号。为此，我们仅基于正常训练图分别构建两个分量的经验分布，将原始
> 分数转换为正常分位数后取最大值。该策略将三次运行的 AUROC 标准差从 0.5489
> 降至 0.0080，说明其显著缓解了评分分量失效导致的随机种子不稳定性。

ADFA-LD 可以写：

> 针对图级读出可能稀释短系统调用异常片段的问题，我们在 HRA-GNN 的图级 SVDD
> 分数之外，引入系统调用 unigram 最近正常距离和三阶 Markov 平均负对数似然。
> 三个分量均以正常训练集经验分位数统一量纲，再以 1、0.5 和 0.25 的权重融合。
> 在固定 ADFA-LD-1000 协议下，完整评分获得 0.8584 AUROC 和 0.8475 AP；
> 移除任一序列分量都会降低 AP。

不能写：

```text
ECDF-Max 在所有数据集上都优于论文评分。
HRA-GNN 原模型无需任何改动就在 ADFA-LD 上达到 0.8475 AP。
HRA-GNN 在 ADFA-LD 两项指标上都相对领先超过 5%。
```

## 16. 审查时建议按此顺序核对

1. 检查 `hra_gnn/model.py::anomaly_score`，确认原论文乘法公式；
2. 检查 `hra_gnn/trainer.py::evaluate`，确认分别保存 SVDD 和 SSL 分量；
3. 检查 `hra_gnn/rescoring.py`，确认 FlowGraph ECDF 只使用正常训练分数；
4. 检查 `hra_gnn/preprocessing.py::prepare_adfa_ld`，确认 syscall 序列与节点
   顺序的对应关系；
5. 检查 `hra_gnn/adfa_scoring.py`，确认 TF-IDF、Markov 和三个 ECDF 的拟合
   数据均为正常训练图；
6. 检查 `configs/splits/adfa_ld_fixed_test_1000.txt`，确认所有方法使用相同图；
7. 检查 `reference_results/adfa_ld_hybrid_ablation_seed9.csv`，确认消融来自
   同一 checkpoint 和同一测试子集；
8. 检查最终表注，确认披露固定子集、序列增强评分和不相等的 seed 搜索预算。

相关结构化结果：

```text
reference_results/adfa_ld_fixed1000_best.csv
reference_results/adfa_ld_hybrid_ablation_seed9.csv
reference_results/adfa_ld_hra_seed9_hybrid_metrics.json
reference_results/hra_seed_sweep_all_runs.csv
```
