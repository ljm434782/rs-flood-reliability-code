# SMAGNet 三分类适配审计（G3 依赖；只读）

**日期：** 2026-09-29  
**结论状态：** 尚未实现、尚未训练、尚未下载作者 checkpoint、尚未读取本项目的 calibration、internal-test、holdout 或外部测试数据。本文不将论文的任何基准数值与本项目结果比较。

## 1. 审计来源、版本与许可

| 项目 | 已核实内容 |
|---|---|
| 论文 | Lee & Li, *A Spatially Masked Adaptive Gated Network for Multimodal Post-Flood Water Extent Mapping using SAR and Incomplete Multispectral Data*, [arXiv:2601.00123v1](https://arxiv.org/html/2601.00123), 2025-12-31；作者页也引用 ISPRS JPRS 2026 版本。 |
| 官方代码 | [ASUcicilab/SMAGNet](https://github.com/ASUcicilab/SMAGNet)，本地只读浅克隆的 `main` HEAD 是 `4371df08e6ca3b9d71c0385ad57b589830469a0c`（2026-01-07T14:40:21-07:00）。 |
| 冻结文件哈希 | `src/smagnet.py`: `0F2BC65BA0B163C475772BBB3171A818E3947C29FBC5EC54F7BC5C479E4A4A58`; `environment.yaml`: `0AAC98B435C65A461E4EABED007E25E8AE536B46A404A010B4F9EC5440907E4F`; `LICENSE`: `67B44FEFA9E1D4EF9D71EA00097DBDC4E556D2EE0EEC755F496051E3B9A75855`. |
| 代码许可 | 仓库的 [`LICENSE`](https://github.com/ASUcicilab/SMAGNet/blob/4371df08e6ca3b9d71c0385ad57b589830469a0c/LICENSE) 为 MIT，允许修改和再分发，须保留版权及许可文本。 |
| 作者 checkpoint | README 链接至 Google Drive，但仓库中没有 checkpoint，且没有单独的权重许可文件。checkpoint 的许可、准确文件版本和哈希均为 **UNVERIFIED**；本项目不应使用它来初始化或比较，除非 G3 单独冻结。 |

只读取了官方仓库的模型、README、依赖文件和示例 notebook 的**代码单元**；没有打开仓库附带的影像/标签样例，也没有打开任何本项目测试数据。

## 2. 官方方法能够确认的接口

### 2.1 输入、时间和空间掩膜

论文使用后洪水 Sentinel-1/2 配对：S1 为 VV/VH 两通道，S2 原始有 13 波段但实际输入 Red、Green、Blue、NIR 四波段；影像都在洪灾发生后四天内获取、10 m 分辨率。论文没有给出一个可直接迁移到本项目的“以 S1 为时间锚，S2 允许早于 S1 若干天”的配对规则。因而项目的单次灾后 S1 和可选单时相 S2 输入在模态形状上可匹配，但时间语义不是对论文方案的逐字复现，必须作为适配范围记录。

官方 notebook 以训练集均值/标准差分别标准化 2 个 SAR 与 4 个 MSI 通道；从加载的 MSI 数组取索引 `[3, 2, 1, 7]` 作为 R/G/B/NIR。它在选带、标准化前将“所有 MSI 原始波段均为 0”的像元记为 invalid。模型把传入的 invalid mask（invalid=1）下采样到每层 gate 分辨率，并令这些像元的 MSI gate 为 0。

项目已有输入为单灾后 S1 的 2×256×256 与单时相 S2 的 12×256×256，并从 SCL、cloud、NaN、数据掩膜及 `-9999` 构造 optical-valid mask（`src/flood_reliability/impactmesh_io.py`）。256×256 满足官方 ResNet encoder 的整步长尺寸要求。可行的适配不是重建官方的“全波段为零”无效规则，而是在冻结的四波段选择后向 SMAGNet 传入项目 valid mask 的逻辑反码（invalid=1）。这会保持项目对云、nodata 和 shared Ignore 的语义，但属于受控适配。

### 2.2 模型和输出语义

官方模型有两个 encoder、各尺度的 1×1 sigmoid gate、一个权重共享 decoder，并返回两条 logits 路径：SAR–MSI fused path 和 SAR-only path。论文说明两条路径各计算 BCE，再以 0.5/0.5 合计。官方 notebook 构造 `classes=1`，对 fused logit 使用 sigmoid，并从 checkpoint 文件名读取在 validation 集确定的 hard threshold。因此作者公开的可执行接口是**二元 water/non-water**，不是三分类洪水/永久水体/non-water。

论文写作称两条 encoder 为 ResNet-50；官方 notebook 也明确用 `encoder_name='resnet50'`。但 `SMAGNet.__init__` 的默认值是 `resnet34`。这不是可忽略的实现细节：适配必须在 G3 冻结 ResNet-50 或 ResNet-34，不能依赖默认值。

`classes` 是模型构造参数，segmentation head 的输出通道随之变化，所以 `classes=3` 在张量形状上是可行的。模型只有一个共享的 segmentation head，即从 1 类到 3 类的参数差为 **290**（最后一层 3×3 convolution 从 145 变为 435 个参数）。整个官方 ResNet-50、三分类模型的总可训练参数量为 **UNVERIFIED**：官方仓库没有给出数值或 checkpoint，本地环境也没有其固定的 `segmentation-models-pytorch==0.5.0` 依赖可供无歧义实例化。G3 应在冻结依赖后，用 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 记录精确值和构造日志；不得用论文或网络估计替代该值。

## 3. 官方训练方案与本项目公平性差异

| 方面 | 官方论文/仓库 | 本项目若作公平三分类适配 |
|---|---|---|
| 标签/输出 | 1 logit、二元 BCE、validation threshold | 3 logits、masked 三类 CE；项目 Ignore=255 对两条输出路径一致生效；argmax，不导入二元阈值。 |
| 主损失 | 两条路径各 BCE，0.5/0.5 相加 | 需要冻结两条路径的 0.5/0.5 masked CE，及是否包含项目的 Dice 项。保留原 BCE 不可处理三类互斥标签。 |
| 输入 | 2 S1 + 4 S2（RGB/NIR） | 同一单时相 2 S1 + 从项目 12 S2 明确映射出的 4 波段；使用同一有效像元掩膜。波段索引映射尚为 **UNVERIFIED**，不得按数组位置猜测。 |
| 缺失语义 | S2 invalid mask 直接抑制 gate；论文另在测试 MSI 中替换缺失像元 | 训练与评估时向 mask 传入项目 invalid；整模态 dropout 时须同时置 MSI 为零、invalid mask 为一；局部遮挡时以同一冻结局部 mask 更新二者。 |
| 归一化 | 训练数据 band-wise mean/std | 只能由项目 fit 过程计算 2+4 通道统计量；不能复用论文数值、开发/校准/测试统计或作者 checkpoint。 |
| 数据划分 | 按 acquisition location 分层随机 6:2:2 | 过程级互斥的 fit/development/calibration/confirmation 角色，不能复用其划分。 |
| 训练 | Adam、lr 5e-4、weight decay 0、batch 16、200 epoch、随机 crop/flip；论文称 10 次重复 | 当前 G2 共同预算为每 seed 24 epoch、65 fit 过程、8 图块/过程/epoch、batch 4、三共同 seed。若纳入，SMAGNet 必须遵守这个预冻结共同预算或在 G3 记录并论证不可避免的例外。 |
| 预训练 | 论文称 SAR encoder 随机初始化、MSI encoder ImageNet 初始化；官方 notebook 两者设为 `None` | 二者不一致。冻结是否使用 ImageNet、权重来源、许可、下载哈希及对所有候选的公平性规则；在此之前为 **UNVERIFIED**。 |

论文给出学习率、batch、epoch 和随机 crop/flip，也称进行 10 次重复；但官方仓库仅有模型定义和推理 notebook，**没有** dataset loader、训练脚本、采样器、augmentation 实现、scheduler、早停/checkpoint 规则、随机 seed 清单或作者训练配置。上述缺项不能由论文结果数字反推。

## 4. 需要实现但尚未执行的最小代码差异

下面是未来 G3 冻结后才可执行的实现清单，不是本轮代码修改：

1. 将官方 `smagnet.py` 以其 commit 和 MIT 许可固定到独立第三方来源目录；记录源码哈希，不覆盖 `data/raw/` 或原项目模型。
2. 建一个项目 adapter，在读取器输出中显式冻结 12→{R,G,B,NIR} 的名称映射，并只保留同一张选定 S2。若字段无法证明波段顺序，停止适配。
3. 使用项目 fit-only 2+4 通道归一化统计。向 gate 传入 `~valid` 的单通道 float mask；整模态和局部遮挡一致更新 optical tensor 与 mask。
4. 构造 `SMAGNet(encoder_name=<冻结值>, classes=3, ...)`，将两条输出都送入相同的 `ignore_index=255` 三类损失；删除 validation-derived hard threshold。需要事先决定两头 0.5/0.5 权重与 Dice 是否为零，不能在开发结果后选择。
5. 接入现有过程均衡采样、三共同 seed、24 epoch 固定末 checkpoint、训练清单/哈希、XPU warning 记录和无测试选择规则；B4/B5/P 与 SMAGNet 使用同一原始配对抽样、状态 mask、Ignore 和确认性评价管线。
6. 在 G3 许可的 fit/development 技术预检中，仅记录实例化参数量、一次 batch 的 XPU 内存、实际 minutes/epoch、checkpoint 大小和数值完整性。该预检不得查看 confirmation 角色或用开发分数调整架构。

## 5. 计算与可行性判断

官方环境锁定 Python 3.9、PyTorch 2.0.1 和 CUDA 11.7；本机为 PyTorch XPU 环境，且当前环境未安装官方源码所依赖的 `segmentation-models-pytorch`/`timm`。因此官方 CUDA environment 不能直接视为可在本机运行的环境。官方实现使用两个 ResNet encoder，显著大于当前 20 万级参数的项目 B4/B5，但没有可核实的官方总参数数、XPU 内存或 epoch 时间。

故三 seed SMAGNet 的 XPU 时间、内存和磁盘数值都是 **UNVERIFIED**，不能纳入 23 XPU h 的已确认预算。适用的唯一规划式为：`3 × 24 × r` 分钟与 `3 × (24 × c + c_selected + logs)` 字节，其中 `r`、`c` 必须由第 4 节所述、冻结环境中的技术预检测得。若预检无法在 XPU 上运行、超出 G3 资源上限，或不能满足相同输入/Ignore/过程角色/状态 mask/训练预算，SMAGNet 不进入 G4，保留为相关工作。

## 6. 审计判定

**技术上可作三分类适配，当前不具备直接运行资格。** 适配可以保持相同的单次 S1 + 可选单时相 S2 输入、共享 Ignore、过程角色、状态 mask 与确认性评价；但它会从官方二元 BCE+阈值任务改变为三类 masked CE+argmax，并替换波段、mask、时间配对、划分、预算和运行环境。结果应称为“在本项目协议下的 SMAGNet-style 三分类适配”，不能称为对作者已发表模型的原样复现，也不得与论文数据集上的基准数值直接比较。

G3 只有在波段名称映射、encoder/预训练选择、两头三类损失、局部遮挡接口、依赖锁定、精确参数量、XPU 资源预检和作者 checkpoint 是否排除均冻结后，才可批准它作为条件性强基线。任一项不能核实，则停止直接对照资格。
