# Sen1Floods11 官方 hand test 与候选主数据的重合审计

日期：2026-09-26。用途：外部测试集独立性核查；**尚未对模型运行外部测试**。

## 来源与技术检查

按照 [Sen1Floods11 官方仓库](https://github.com/cloudtostreet/Sen1Floods11) `v1.1/splits/flood_handlabeled/flood_test_data.csv`，从官方 GCS bucket 取得 90 对 S1Hand/LabelHand 栅格及两份元数据。182 个对象逐个核对 GCS MD5 并记录 SHA256；清单见 `data/manifests/sen1floods11_manifest.csv`。90 对来自 10 个事件；标签值为 `-1/0/1`，S1 与标签的网格差异均低于 0.001 像元。逐场景审计见 `data/manifests/sen1floods11_test_audit.csv`。`-1` 作为 Ignore，不参与最终 water/non-water 指标。

## 与原始 STURM 的事件及空间重合

先对照官方 Sen1Floods11 事件地理元数据和原始 STURM S1/S2 CSV 的国家。10 个外部测试事件中，仅 Spain 与 Pakistan 出现在 STURM 任一模态的国家列表；官方 Sen1 元数据中 `Mekong` 对应 Cambodia。再从官方 STURM ZIP 选择性提取 Spain/Pakistan 的所有 2,233 个掩膜：S1 1,794 张（Spain 705、Pakistan 1,089），S2 439 张（Spain 397、Pakistan 42）。将每张栅格的地理包围盒转换为 EPSG:4326，与 90 张外部测试标签的包围盒逐张检查。

**结果：90 张外部测试图块中，零张与所检查的同国家 STURM S1 或 S2 图块的地理包围盒相交。** Spain 外部事件与 STURM 同国最近 S1/S2 获取时刻分别相差约 35.3/128.4 天；Pakistan 分别相差约 1,889/1,888 天。其它 8 个外部事件在 STURM 两套元数据中没有同名国家。这些时间差使用 Sen1 事件级、仅精确到日的 `s1_date` 计算，属于近似值。逐图块结果见 `data/manifests/sen1_sturm_event_overlap_audit.csv`。

这一结果支持：在所用国家字段范围内，外部手标测试图块与同国 STURM S1/S2 瓦片没有直接空间重叠。按国家预筛未检查标为邻国、但实际跨境覆盖外部图块的瓦片；包围盒不相交也不能证明两个不同地点不受同一大范围洪水过程影响。最终训练资产清单冻结后，须不依赖国家字段，将实际入选的全部训练影像范围与 90 张外部测试图块比对，并检查共同洪水过程。若主数据换成 ImpactMesh，也必须重新做这项检查。正式外部测试仍在训练和校准完成后才执行，不用这 90 对图块选择模型、阈值或筛选规则。

复算：`python scripts/index_sturm_zip.py --external-overlap-country-masks`，`python scripts/audit_external_event_overlap.py`。

### 不按国家预筛的近期同分带补充核查

为进一步检查上述国家字段局限，另以全部 STURM S1/S2 原始元数据为输入，对 90 张外部图块分别筛选：卫星获取时间距其事件级 `s1_date` 不超过 30 天，且栅格 `epsg_code` 属于外部图块经度所在 UTM 分带及左右各一带、同半球。这是**保守的时空候选初筛**，不是最终过程独立性的定义。复算脚本为 `python scripts/audit_sturm_external_time_zone.py`。

筛查只留下 **227 张 S1 掩膜**，全部是标为法国的 EMSR324，关联 Ghana 11 张和 Nigeria 4 张外部图块；没有 S2 候选。已从 STURM 官方 ZIP 逐件读取这 227 张掩膜，利用 ZIP 成员 CRC/大小核验，并另存 SHA256 与实际投影范围。它们实际位于经度 2.2822–3.1373、纬度 43.1406–43.3259；Ghana 外部图块位于纬度 8.2789–10.6705，Nigeria 位于 9.0608–9.8887。因此 **15 张近期同分带外部图块与这些实际掩膜均无地理包围盒交集**。其余 75 张图块在这两个预筛条件下没有 STURM 候选。逐掩膜记录与逐场景结果分别见 `data/manifests/sturm_external_time_zone_masks.csv` 和 `data/manifests/sturm_external_time_zone_audit.json`。

这一补充检查没有下载、逐件读取全部 STURM 掩膜；它依赖原始 CSV 的时间和 EPSG 字段，以及所选的 ±30 天、相邻 UTM 分带范围。相隔更久或元数据错误的共同过程、邻近但不重叠的瓦片、未来正式训练中可能增加的资产，仍需在冻结实际入选清单后核查。不能把本次零交集写成全数据集的外部独立性证明。

## 与 ImpactMesh 官方清单的空间筛查

**方法修正（2026-09-26）：**原筛查按 24,578 个样本 ID 的 MGRS 带推算图块 CRS，仅核对四张本地标签。后续固定训练试点 rank 14 显示样本 ID 的 `32TQQ` 与真实 TIFF 图块的 EPSG:32633 不一致。现已完整下载并按发布方 LFS SHA256 校验 train/val/test 三份官方 `MASK.tar`（分别 48,650,240 / 5,877,760 / 7,608,320 字节）；运行 `python scripts/audit_impactmesh_mask_grids.py --split train|val|test`，逐件只读取 **24,578 张 TIFF 的 CRS、transform 与边界头部，不读取标签像元**。样本 ID 的 UTM 带与真实网格 CRS 不一致者分别为 train **870/19,448**、val **424/2,171**、test **132/2,959**；但全部图块 ID 的 `x/y` 与 TIFF 左上像元中心坐标一致。逐件机器清单为 `impactmesh_{train,val,test}_mask_grid_index.csv` 与配套元数据；28 张本地已下载标签的 CRS/边界与归档头部索引一致。旧的 ID 推算结论已被本次实栅格头部复算替代，不得再引用其地理位置依据。

现运行 `python scripts/audit_impactmesh_sen1_overlap.py`，从真实 TIFF CRS/边界构造 24,578 个多边形，再与 90 张 Sen1Floods11 手标测试图块逐一相交。来源清单和网格索引 SHA256 记于 `data/manifests/impactmesh_sen1_overlap_audit.json`。这一筛查读取外部标签的地理参考信息，不读取其像元值。

实栅格头部重算后，交叠数量碰巧与旧估计一致：官方 train 的 19,448 张、val 的 2,171 张与 90 张外部测试图块无直接面积重叠；普通 test 的 2,959 张中，有 **4 张 EMSR388 图块与 3 张 Spain 外部图块交叠**（7 对图块关系）。官方 `test_holdout` 562 张无直接面积重叠；它是普通 test 的子集。**训练集零直接空间相交不代表洪水过程独立**：EMSR388 同时在 ImpactMesh 官方 train 中有 113 张、普通 test 中有 6 张，官方拆分本身也不能当成 EMSR388 的过程隔离。外部 Spain 测试共 6 张，官方事件元数据的 S1 日期是 **2019-09-17**；[CEMS 官方 EMSR388 记录](https://mapping.emergency.copernicus.eu/activations/EMSR388/)把它列为 **2019-09-12 启动的西班牙东南部洪水**。空间和时间证据支持把这 6 张 Spain 外部图块整体列为**疑似同一洪水过程**，在最终外部过程独立评价中单列或排除，不能仅删除 3 张直接相交图块后宣称其余 Spain 图块独立。是否确属同一过程仍需最终入选样本的影像时刻及 CEMS 产品链核对。

这一核查不证明原始 Sentinel 影像文件相同，也不能因其它图块没有面积重叠就断言洪水过程完全独立。若最终选择 ImpactMesh 为主数据，应在预注册前决定 EMSR388 与 Spain 外部测试的处理方式，并针对**最终入选的训练/验证/校准/测试资产**重算一次，不使用外部测试标签结果选择规则。

### 官方未见过程留出集的空间距离初筛

为辅助判断路径 B 的官方 `test_holdout` 是否还有邻近外部事件，运行 `python scripts/audit_impactmesh_holdout_external_distance.py`。脚本用这 562 张官方留出图块的**真实 TIFF CRS/边界**，并读取 90 张 Sen1Floods11 官方手标测试标签的地理范围；没有读取其标签像元。按 12 个 ImpactMesh activation 分组，对 10 个 Sen1 事件逐图块计算 EPSG:6933 等面积投影中的最小多边形间距。最近一组仍是 EMSR844 与 Mekong，诊断距离约 **955.153 km**；其余 11 组更远。旧 ID 推算法将 EMSR358 到最近外部事件的距离报为 1,398.611 km，实网格头部修正为 **1,929.522 km**，说明旧距离明细不可复用。每组的最近事件、图块 ID、距离与输入清单 SHA256 保存在 `data/manifests/impactmesh_holdout_external_distance.json`。

该数值是投影距离诊断，**不是精确大地测量距离**。它支持“这 12 个官方留出 activation 与当前 90 张外部测试图块相距较远”的空间判断；不能证明共同洪水过程、影像来源或全量数据的独立性。前述 EMSR388/Spain 风险仍适用于官方 train 和普通 test，且正式研究若重分过程，必须对最终入选资产重新审计。

### 2026-09-27 最终训练角色与外部 Nigeria 事件复核

正式 B1 外部评估已写出但**尚未查看数值汇总**时，进一步以 Sen1Floods11 官方事件元数据（只含几何与日期）对照最终 ImpactMesh 训练角色所有 CEMS activation。发现外部 Nigeria 的官方 S1 事件日为 **2018-09-21**，而训练 `EMSR314` 的 45 张 S1 时间为 2018-09-21 至 22 日，其中 32 张进入正式配对训练。四张外部 Nigeria 图块与 `EMSR314` 训练 TIFF 足迹的最近 EPSG:6933 投影边界间距分别约 **118.1、66.0、95.4、19.6 km**；零面积相交的旧筛查不足以证明不同洪水过程。[CEMS EMSR314 官方记录](https://mapping.emergency.copernicus.eu/activations/EMSR314/)和[官方新闻](https://mapping.emergency.copernicus.eu/news/copernicus-ems-monitors-floods-in-nigeria/)描述 2018-09-17 起 Niger/Benue 河洪水，与外部 Nigeria 时间和地理范围高度一致。故将 Nigeria 整个外部事件与预先隔离的 Spain 一起排除独立主汇总，两者仍保留单列的关联事件敏感性结果。排除清单为 `data/manifests/sen1_external_process_exclusions.csv`。

这是**B1 外部像元已读取、但未查看该结果数值且其余四模型外部测试尚未运行时的证据驱动修订**。原始 B1 输出在重算前另存以保留审计痕迹；所有五模型按相同的 Spain/Nigeria 修订规则生成最终外部主汇总。外部 Ghana 与训练 Nigeria 同期但相距约 873 km，现有证据不足以认定为同一河流洪水过程；其独立性仍非绝对证明。
