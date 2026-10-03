# ImpactMesh 标签时间链核查

状态：**逐样本来源审计进行中**。此文件区分卫星参考影像获取时刻、CEMS 产品交付时刻、ImpactMesh 事件 S1 获取时刻及数据集掩膜的实际产品映射。相近的时间或空间形状不能单独证明发布方采用了某个 CEMS 产品。

## 固定训练试点 rank 1：EMSR156 AOI01

ImpactMesh 样本 `EMSR156_1_29UNV_x545285_y5951505` 属于预先固定的训练来源试点。事件索引（2）S1 获取于 **2016-03-09 18:22:42.643994 UTC**；256×256 标签中有 1,565 个洪水类（2）像元、15 个永久水体类（1）像元。[ImpactMesh 数据卡](https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood)只称标签来自 CEMS，没有提供这张图块对应的 CEMS 产品 ID。

[CEMS EMSR156 activation 页面](https://mapping.emergency.copernicus.eu/activations/EMSR156/)只列出两张 delineation overview 产品，AOI01 Castelplunket 和 AOI02 Lough Funshinagh。已下载并目视核查两张[官方地图：AOI01](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR156/EMSR156_01CASTELPLUNKET_DELINEATION_OVERVIEW_v1_200dpi.pdf)、[AOI02](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR156/EMSR156_02LOUGHFUNSHINAGH_DELINEATION_OVERVIEW_v1_200dpi.pdf)及各自矢量包；来源 URL、文件 SHA256 与参考影像时刻记于 `data/manifests/cems_index.csv`。地图图例和数据来源区均标注洪水范围参考影像为 **RADARSAT-2，2016-03-09 18:32 UTC**，比样本的事件 Sentinel-1 影像晚 **9.289 分钟**。这是近时但**不是同一卫星或同一获取时刻**；产品交付发生在次日，不等于影像参考时刻。

运行 `python scripts/audit_emsr156_label_chain.py`，可从两份已校验 SHA256 的官方矢量包读取 AOI 和洪水多边形，将其投影到 ImpactMesh 标签网格，并按像元中心栅格化。结果保存在 `data/manifests/emsr156_train_label_chain.json`：

| 官方产品 | 图块位于 AOI 的面积比例 | 图块内官方洪水像元 | ImpactMesh 洪水类与官方洪水交集 | 标签仅有 / 官方仅有 | Flood IoU |
|---|---:|---:|---:|---:|---:|
| AOI01 Castelplunket | 1.000 | 1,459 | 1,444 | 121 / 15 | 0.914 |
| AOI02 Lough Funshinagh | 0.000 | 0 | 0 | 1,565 / 0 | 0 |

AOI01 官方洪水范围多出的 15 个像元在 ImpactMesh 中均为永久水体类。AOI01 的位置、近时参考影像和高空间重合**强烈支持**该产品为本样本洪水类别的来源；差异像元可能来自栅格化或数据集再处理，不能据此声称原始标签与官方矢量逐像元完全一致。发布方未提供这张图块到产品版本的直接映射，因此来源结论保留为“高度吻合，未直接证实”。该例不能外推至其余 23 个试点 activation，更不能证明所有标签都与事件 S1 同步。

## 固定训练试点 rank 3：EMSR640 AOI02

按固定清单顺序跳过洪水类像元为零的 rank 2，选取下一张含洪水类的图块 `EMSR640_2_30VWJ_x547805_y6332575`。其事件 S1 获取于 **2022-11-20 06:30:34.489545 UTC**，标签洪水类有 1,995 像元。图块完整位于 [CEMS EMSR640 activation](https://mapping.emergency.copernicus.eu/activations/EMSR640/) 的 AOI02，与其余三个 AOI 无交集。已下载四个 AOI 的基础矢量包以定位图块，以及 AOI02 的基础、MONIT01、MONIT02 地图和矢量包；来源、SHA256 和图件标注的最新参考影像时刻见 `data/manifests/cems_index.csv`。

三张[基础图件](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR640/EMSR640_AOI02_DEL_PRODUCT_r1_RTP01_v3.pdf)、[MONIT01 图件](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR640/EMSR640_AOI02_DEL_MONIT01_r1_RTP01_v3.pdf)、[MONIT02 图件](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR640/EMSR640_AOI02_DEL_MONIT02_r1_RTP01_v3.pdf)已目视核查。矢量的 `observedEvent` 需按 `notation` 区分当期 `Flooded area` 与此前 `Previous flooded area`。运行 `python scripts/audit_emsr640_label_chain.py` 会校验本地地图、矢量和标签的 SHA256，再将多边形按像元中心栅格化到标签网格。机器结果见 `data/manifests/emsr640_train_label_chain.json`：

| AOI02 范围 | 图件最新参考影像 | 与事件 S1 的时差 | 洪水类 IoU | 标签仅有 / 官方仅有 |
|---|---|---:|---:|---:|
| 基础图件当期 | Sentinel-1A，11 月 18 日 17:59 UTC | 早 36.53 小时 | 0.391 | 1,132 / 213 |
| MONIT01 当期 | Sentinel-1A，11 月 20 日 06:30 UTC | 图件时间早 34.49 秒 | **0.873** | 254 / 0 |
| MONIT02 当期 | COSMO-SkyMed，11 月 21 日 18:11 UTC | 晚 35.67 小时 | 0.289 | 1,318 / 346 |
| MONIT02 此前范围 | 沿用 MONIT01 的洪水范围 | 参考时刻应追溯 MONIT01 | **0.873** | 254 / 0 |

基础图件当期范围与 MONIT01 的此前范围在此图块**逐像元相同**；MONIT01 当期范围与 MONIT02 的此前范围也**逐像元相同**，均由脚本明确比较。故不能仅因 0.873 的 IoU 就声称标签取自 MONIT01 这一份产品文件：MONIT02 同时保留了相同的此前范围。不过，吻合的**洪水态势时刻**与 MONIT01 图件 11 月 20 日 06:30 UTC 的 Sentinel-1A 参考影像相符，而非 MONIT02 新增的 11 月 21 日当期范围。图件只精确到分钟，34.49 秒的差值不能证明两个记录对应同一卫星产品。上述证据支持该标签反映接近事件 S1 的洪水态势，**不构成发布方逐图块产品映射或整套数据同步性的直接证明**；图件交付日期也不能当作影像获取日期。

## 固定训练试点 rank 4：EMSR424 AOI01

下一张含洪水类的固定训练图块是 `EMSR424_1_38LQH_x786235_y8282875`。事件 S1 获取于 **2020-01-31 02:26:50.969885 UTC**，标签含 29,523 个洪水类（2）像元、1,352 个永久水体类（1）像元；全图块位于 [CEMS EMSR424 activation](https://mapping.emergency.copernicus.eu/activations/EMSR424/) AOI01。已下载、校验 SHA256 并目视核查官方[首次估计图](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR424/EMSR424_AOI01_FEP_PRODUCT_r1_RTP01_v2.pdf)、[基础 delineation 图](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR424/EMSR424_AOI01_DEL_PRODUCT_r1_RTP01_v2.pdf)、[MONIT01 图](https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR424/EMSR424_AOI01_DEL_MONIT01_r1_RTP01_v1.pdf)及三份对应官方矢量包；六件文件的 URL、SHA256、图件所列最新参考影像时刻在 `data/manifests/cems_index.csv`。首次估计采用 1 月 27 日 06:54 UTC 的 Sentinel-3A/B；基础图采用 1 月 31 日 02:26 与 02:27 UTC 的 Sentinel-1A；MONIT01 在此基础上还采用 2 月 3 日 07:11 UTC 的 Sentinel-2B。三图的制作日期晚于各自参考影像时刻，不能替代标签时刻。

运行 `python scripts/audit_emsr424_label_chain.py`，会核对地图、矢量和标签哈希及标签类别计数，读取各矢量的 `observedEventA`，仅将 `notation=Flooded area` 多边形按像元中心栅格化到标签网格。`Flood trace` 不计入洪水类。逐像元交并与 AOI 覆盖保存在 `data/manifests/emsr424_train_label_chain.json`：

| AOI01 官方矢量 | 图件最新参考影像 | 洪水类 IoU | 标签仅有 / 官方仅有 |
|---|---|---:|---:|
| 首次估计 FEP | 1 月 27 日 06:54 UTC，早于事件 S1 91.55 小时 | 0.198 | 17,663 / 30,518 |
| 基础 delineation | 1 月 31 日 02:26 和 02:27 UTC，跨越事件 S1 的分钟级时刻 | **0.961** | 1,139 / 0 |
| MONIT01 合并 `Flooded area` | 图件最新为 2 月 3 日 07:11 UTC，比事件 S1 晚 76.74 小时 | **0.961** | 1,139 / 0 |
| MONIT01 来源 ID 3 的此前范围 | 图例对应 1 月 31 日的既有洪水范围 | **0.961** | 1,139 / 0 |
| MONIT01 来源 ID 5 的当期范围 | 图例对应 2 月 3 日的新增判读 | **0.701** | 8,829 / 0 |

基础图与 MONIT01 **合并后**的 `Flooded area` 在这张图块内栅格化后相差 0 像元；首次估计与基础图相差 48,656 像元。MONIT01 的来源 ID 3 此前范围在图块内为 28,384 像元，与基础图相同；来源 ID 5 当期范围为 20,694 像元，全部落在 ID 3 内，合并掩膜因而没有增加像元。来源 ID 4 在本图块内为零像元。**合并掩膜相同不代表 MONIT01 没使用三天后的 S2**；图件确实将部分范围标为 2 月 3 日当期判读。也不能把 MONIT01 图件最新影像时刻直接当成全部多边形的参考时刻。

高重合支持标签反映接近 1 月 31 日 S1 的洪水范围，不能从这张图块唯一辨认所用 CEMS 产品文件。基础图的两次 S1 获取只标到分钟，无法确定 ImpactMesh 事件 S1 对应哪次官方影像；索引中基础图的 02:27 是**图件最新**参考影像时刻，不是对每个洪水多边形的单独时刻。1,139 个标签独有洪水像元的处理来源未明。发布方未给出逐图块产品映射，因此当前为“洪水态势高度吻合，具体产品和成图链未直接证实”；这不是其余 23 张训练试点或全量标签的同步性结论。
