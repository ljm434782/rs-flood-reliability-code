# SMAGNet 三分类适配与合成 XPU 技术审计

**状态：仅技术冒烟；不是训练、模型选择、校准或结果。**

## 上游来源与边界

- 作者 MIT 源码现锁定在稳定路径 `third_party/smagnet`，commit 为
  `4371df08e6ca3b9d71c0385ad57b589830469a0c`。
- `src/smagnet.py` 的 SHA-256 为
  `0f2bc65ba0b163c475772bbb3171a818e3947c29fbc5ec54f7bc5c479e4a4a58`；MIT
  `LICENSE` 的 SHA-256 为
  `67b44fefa9e1d4ef9d71ea00097dbdc4e556d2ee0eec755f496051e3b9a75855`。
- `flood_reliability.smagnet_adapter` 不修改或复制作者源码。它需要调用者提供
  源码路径，并在导入前验证上述 SHA-256；内容不匹配时失败。
- G3 冻结时以 `scripts/fetch_smagnet_source.py --destination third_party/smagnet`
  从 `https://github.com/ASUcicilab/SMAGNet.git` 获取该 commit，写入
  `smagnet.py`、MIT `LICENSE` 与 `SOURCE.json`。脚本拒绝覆盖已有目的地，并在写入前
  校验源码和许可证哈希；训练配置将指向此稳定、非 `tmp` 路径。
- 本适配将作者的两个输入分支固定为 `[VV,VH]` 与 `[B04,B03,B02,B08]`，并将
  `classes=3`。输出仍是作者的融合头和 SAR 头。无预训练权重。
- 已执行该取得脚本，稳定来源为 `third_party/smagnet`：`smagnet.py` 和 `LICENSE`
  分别复核为上述两个 SHA-256，`SOURCE.json` 的 SHA-256 为
  `496a8615ffe5fd5a7e133c87af35a5df80fbcb8d5d4c2a0550d5afd1b367b9eb`。

## 当前本机依赖

未升级或卸载既有 PyTorch。实际导入版本为：

| package | version |
| --- | --- |
| torch | `2.14.0+xpu` |
| torchvision | `0.29.0+xpu` |
| segmentation-models-pytorch | `0.5.0` |
| timm | `1.0.24` |
| safetensors | `0.8.0` |
| tensorboard | `2.21.0` |

完整的 G3 运行时版本锁与安装后 METADATA SHA-256 位于
`requirements/smagnet_g3_xpu.lock.txt`（该锁自身 SHA-256 见下方工件表）。当前环境没有保留
这些发行版的原始 wheel 文件，故**所有** wheel artifact hash 均为 **UNVERIFIED**；锁记录了
当前已导入发行版的精确版本与 METADATA hash，且本审计未升级或替换 torch。

安装先加入 XPU 兼容的四个直接依赖，随后因作者源码在导入时引用
`torch.utils.tensorboard` 而加入 `tensorboard` 及其小型传递依赖；未替换 torch：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps torchvision==0.29.0+xpu segmentation-models-pytorch==0.5.0 timm==1.0.24 safetensors==0.8.0 --extra-index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe -m pip install tensorboard
```

## 合成检查定义

`tests/test_smagnet_adapter.py` 构造一个 `[1,6,256,256]` 张量、一个 `[1,1,256,256]`
invalid mask，和同时含 `0/1/2/255` 的标签。它检查 ResNet50 三类双头的形状、反向
传播、两头各自的加权 CE + `0.3 *` 类 2 Dice，以及改变 `255` 像元类别不会改变损失。
该检查不读取项目的任何像元、标签、角色清单或推理结果。

无效掩膜的冻结候选契约为 `torch.bool` 或数值型的有限二值张量，形状**严格**
`[B,1,H,W]`，其中 `0` 为有效、`1` 为无效，且 `H,W` 必须与标签完全一致。适配器拒绝
`[B,H,W]`、错误空间尺寸、`NaN`、`0.5` 和其他非二值值；测试覆盖这些拒绝路径。

`scripts/smoke_smagnet_synthetic.py` 只在 XPU 上做一次 forward/backward，明确不创建
optimizer、不调用 `optimizer.step()`、不保存权重或 checkpoint。资源输出写到调用者显式
指定的 JSON 路径；完成后将在本文件记录实际数值和脚本哈希。

## 已执行的 XPU 合成测量

命令为：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_smagnet_synthetic.py --output tmp\smagnet_audit\synthetic_xpu_resource.json
```

在 Intel Arc 140V（16 GB）和 torch `2.14.0+xpu` 上，单个 `[1,6,256,256]` 纯合成
batch 的一次 forward/backward 结果如下：

| measure | value |
| --- | --- |
| parameters | 56,036,248 |
| elapsed forward/backward | 4.8781 s |
| throughput | 0.2050 samples/s |
| peak allocated XPU memory | 625,995,264 bytes (597.0 MiB) |
| optimizer created / step | `false` / `false` |
| weights or checkpoint written | `false` |

该 JSON 的 `synthetic_only` 为 `true`，无数据集读取。此次数值是单次技术资源测量，
不用于训练预算、架构选择或任何科学结论。

批量 `4` 的同一无更新测量为 6.8253 s/batch、0.5861 samples/s、峰值已分配
1,675,004,416 bytes（1,597.4 MiB）。这只包含一次 synthetic forward/backward；没有
optimizer、数据读取、日志或 checkpoint 序列化。按 2,880 updates/run × 3 seed 机械相乘，
该窄测量对应 16.38 XPUh；加入 20% 仅为 19.66 XPUh。因此它不能支持“≤12 XPUh”的正式
训练判断，实际训练仍可能因 optimizer、I/O、日志及保存更慢。

随后完成纯合成的完整训练步测量（`Adam(..., foreach=False)`，一次 `step()`，临时模型且
不写权重/checkpoint）：batch 1 为 5.1702 s 和 975,197,184 B（930.0 MiB）峰值；batch 4
三次为 6.6261、6.4917、6.5747 s/batch，均为 1,675,004,416 B（1,597.4 MiB）峰值。故完整
batch-4 step 的已观测上界是 6.6261 s；其 8,640 updates 的纯计算外推为 15.90 XPUh，
加 20% 为 19.08 XPUh。三次合成测量仍不能提供含 I/O、日志、序列化与硬件争用的可靠
实际训练上界。为避免把窄测量误报为保证，建议 G3 若继续纳入 SMAGNet，将其有限 stop
cap 设为 **24 XPUh**（相对已观测纯计算约 50% 余量），并明确其为管理 cap、不是吞吐保证；
与既有非 SMAGNet 23 XPUh 预约合计为 **47 XPUh**。

## 校准内存门槛

本机物理 RAM 为 31.61 GiB；协调快照记录的空闲 RAM 为 19.61 GiB，复测时为 19.41 GiB
（可随其他进程波动）。若校准阶段同时保留 2,146 个 calibration 样本、五个状态、三类
`256×256` float32 logits，则理论张量容量为 8,438,415,360 bytes（7.8589 GiB），未含标签、
索引、Python 开销或优化器。冻结运行的校准 RAM 门槛应为启动时空闲 RAM **至少 12 GiB**；
低于该值不启动校准，应使用经独立复核的分块/流式实现而不能改变样本或状态。

## 正式 checkpoint 容量策略

本次审计未写入任何项目训练 checkpoint；仅在受限纯合成审计中写入唯一临时完整 payload
并立即删除。对将来的冻结运行，SMAGNet 每个 epoch 仅保留一个原子
`last-complete` checkpoint（用于协议允许的一次硬件恢复）；在下一完整 epoch 成功后删除
前一个。第 24 epoch 完成后保留唯一最终 checkpoint 和哈希，不保留前 23 个。无压缩
FP32 `state_dict` 的精确张量载荷为 224,570,800 bytes（214.2 MiB，668 tensors）；文件
封装开销须在 G3 前实际测量并记录。optimizer 状态只保留在运行期的 `last-complete` payload；
已完成 seed 的 final 为 model-only。该大小测量以纯合成临时模型创建过 optimizer，但没有
保留模型或训练目录，不能把这些数字当作完整训练目录总量。

为核实封装开销，唯一临时目标
`tmp/smagnet_audit/TEMP_DELETE_AFTER_FULL_CHECKPOINT_SIZE.pt` 保存了一个纯合成、已经完成
一次 Adam step 的 `model_state + optimizer_state`，实测为 673,445,733 bytes（642.25 MiB），
SHA-256 为 `921bb7adfa7c5769d4bd2105fd05c7af4fdeb7b2fe8cf010a6b3d38a2d74bbe4`，随后在同一
进程删除并验证不存在。它比张量下界大 584,949 bytes；没有保留模型权重。

同样方式对唯一临时目标
`tmp/smagnet_audit/TEMP_DELETE_AFTER_MODEL_ONLY_CHECKPOINT_SIZE.pt` 测得 model-only payload
为 224,830,321 bytes（214.42 MiB），SHA-256
`57ebf13021a6671a50c3493221f58f7782832594d848fb4976834a1b42d9b6d5`，并在同一进程删除。
因此串行第三个 seed 训练期间的增量磁盘峰值为：两个已完成 model-only final 加两个原子
full `last-complete` 文件，共 1,796,552,108 bytes（1,713.33 MiB）；加上实测新增依赖
53,589,286 bytes 后为 1,850,141,394 bytes（1.72 GiB / 1.85 GB）。这低于 3 GB 增量 cap，
保留约 1.15 GB 给日志、源码和序列化差异；该结论依赖串行执行、每轮仅一个可恢复 checkpoint，
以及完成 seed 仅保留 model-only final。现有逐 epoch 累积 checkpoint 实现不满足此策略。

## G3 候选清单复算与种子检索边界

`scripts/audit_g3_candidate_manifest.py` 已独立重跑。候选仍是
`candidate_only_not_frozen`：fit 3,525 个 ID/60 个单位，calibration 2,146、internal 2,094、
holdout 492、historical development 867；所有角色对的 sample-ID 交集为 0。该复算说明目前
候选角色闭包完成，但其边界仍只是候选清单，尚不能取代 G3 的冻结包。

新 seed `20261013`、`20261014`、`20261015` 的文本检索覆盖 6 个 `configs` JSON、31 个
`outputs/**/run.json` 与 190 个 `data/manifests` 的 JSON/CSV/TXT，均未发现匹配。该检索范围
只覆盖可读文本配置、运行记录和清单；**未检查二进制 checkpoint 的 metadata**，也未读取
影像、标签或模型评分。

## 工件哈希

| artifact | SHA-256 |
| --- | --- |
| `src/flood_reliability/smagnet_adapter.py` | `2e237fffb16cacde60590641f54e81273469129918cf24301b92b6445775705e` |
| `tests/test_smagnet_adapter.py` | `ad7395c2dbdd74a0f9a6bb5d77133b3c6218fffbcc2062c20ac14c8842c809d8` |
| `scripts/smoke_smagnet_synthetic.py` | `bad8b58533e55a2921488603bcc28321a93b3be448742e02a3172384323d1774` |
| `scripts/fetch_smagnet_source.py` | `7dccdac1cc5fda24b550d1cd55f93fbc63503ca226b82a026605fe1fb8944fc2` |
| `scripts/measure_smagnet_temporary_checkpoint.py` | `704677a6b85d9394c98f3cb4daa59d9937ae5a25b6822b09fa3687e8873de47a` |
| `scripts/audit_g3_candidate_manifest.py` | `90677bfdeb83460cdcbf68086287d0a12650759c93775f79f6c888c89e5b6556` |
| `requirements/smagnet_g3_xpu.lock.txt` | `5b6196692effb7df5242b28c58c02887869cb6577d886533d7c25096e3fde4a8` |

## 正式实验前仍需完成的事项

G1 的候选 process-family 角色闭包已完成复算，但边界仍为未冻结候选。正式实验还必须在
冻结角色、冻结输入归一化与掩膜协议后，锁定稳定源码、依赖和运行规则，并按预注册范围运行。
