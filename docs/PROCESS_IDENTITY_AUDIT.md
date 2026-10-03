# ImpactMesh 过程身份审计（正式训练前）

2026-09-27。只使用经 SHA256 校验的官方 train S1 时间元数据、官方 train 标签 TIFF 的 CRS/transform/边界（未读像元）与 [CEMS activation 官方资料](https://mapping.emergency.copernicus.eu/about/how-to-harvest-cems-mapping-data/)。`scripts/audit_impactmesh_process_identity.py` 核对 19,448 张 train 图块的 S1 时间，筛出 S1 时间范围相隔不超过 30 天、足迹中心不超过 500 km 的 37 对 activation；这只是人工核查列表，不能将邻近直接认定为同一过程。原始 API 快照保存在只读 `data/raw/cems/`；标准化表、候选表和 SHA 元数据在 `data/manifests/`。

## 合并证据

| 合并的 activation | 公开事件证据 | 判定 |
|---|---|---|
| [EMSR228](https://mapping.emergency.copernicus.eu/activations/EMSR228/) / [EMSR231](https://mapping.emergency.copernicus.eu/activations/EMSR231/) | 北爱尔兰与西北爱尔兰同期跨境暴雨洪水；S1 范围重叠、足迹中心相距 24.5 km；[JRC 事件记录](https://data.jrc.ec.europa.eu/dataset/d494dfdc-7d4d-47dd-b449-0a2c416826f0) | 同一洪水过程，合并 |
| [EMSR421](https://mapping.emergency.copernicus.eu/activations/EMSR421/) / [EMSR422](https://mapping.emergency.copernicus.eu/activations/EMSR422/) | 法国 Occitanie 与西班牙 Girona 同期洪水；422 明指 Storm Gloria，[Copernicus 气候总结](https://surfobs.climate.copernicus.eu/stateoftheclimate/january2020.php)记载 Gloria 同时影响法国南部 | 同一风暴洪水，合并 |
| [EMSR550](https://mapping.emergency.copernicus.eu/activations/EMSR550/) / [EMSR554](https://mapping.emergency.copernicus.eu/activations/EMSR554/) | 官方 AOI 均包含 Goondiwindi 和 Wee Waa；后者描述南昆士兰与北新南威尔士持续降雨；[CEMS 洪水综述](https://mapping.emergency.copernicus.eu/news/information-bulletin-154-cems-activities-for-australia-floods/) | 保守合并同一持续区域洪水；不声称同一洪峰 |
| [EMSR777](https://mapping.emergency.copernicus.eu/activations/EMSR777/) / [EMSR778](https://mapping.emergency.copernicus.eu/activations/EMSR778/) | 官方两项均命名 Tropical Storm Sara，分别为 Belize/Honduras；S1 相隔约 1 分钟 | 同一命名风暴，合并 |

其余近邻对没有充分同一过程证据。重点复核：`EMSR233/243` 是不同命名的 Irma/Maria；`EMSR472/475` 为越南同地先前洪水和之后登陆的 Molave；`EMSR762/771` 为 Emilia-Romagna 先后两次降雨事件；`EMSR855/858` 为意大利不同地区和气旋；`EMSR567/570` 为澳大利亚相隔约一月的两轮洪水。`EMSR280/283`、`EMSR271/273`、`EMSR329/333`、`EMSR762/768` 在不同国家、流域或时段，邻近本身不构成合并证据。相近地区再访仍可能让模型学习地理特征，不能声称严格地域外泛化。

`scripts/finalize_impactmesh_process_groups.py` 将上述四对作为确定性边，取组内字典序最小 activation 为过程 ID，并按已冻结的哈希顺序重跑分配。输入和输出 SHA256、过程数量见 `data/manifests/impactmesh_process_split_grouped_meta.json`。不同过程 ID 的独立性是根据现有公开元数据审计，不是每一场洪水的水文重建；超过 30 天或 500 km 的同一大范围灾害可能漏检。未来识别到真实跨角色同过程时，主结果须标记受影响，不能在看过测试结果后暗中改清单。

## 2026-09-27 训练启动后的公开事件证据补正（校准/测试标签尚未读取）

最初的 30 天/500 km 地理筛查漏掉跨较远地区的同一**命名风暴**。对官方 CEMS 事件名称和专题公报作不涉及影像/标签的补查，发现 [CEMS Hurricane Irma 官方综述](https://mapping.emergency.copernicus.eu/news/copernicus-ems-is-mapping-the-impact-of-hurricane-irma-over-the-carribbean-islands/)同时列出验证 `EMSR232` 与校准 `EMSR233`。为保持校准与已使用的验证集过程独立，校准主分析排除 `EMSR233`。

[CEMS 2024 中东欧洪水官方公报](https://mapping.emergency.copernicus.eu/news/information-bulletin-173-the-copernicus-emergency-management-service-monitors-floods-in-central-and-eastern-europe/)明确将校准 `EMSR756`、内部测试 `EMSR762`、官方留出 `EMSR757/763/766` 列为 Storm Boris 同一大范围天气洪水的关联 activation。内部测试主分析排除 `EMSR762`；官方留出独立主汇总排除 `EMSR757/763/766`，若生成它们的结果须单列为关联事件敏感性结果。剩余校准和内部测试各 **19** 个过程，官方留出至多 **9** 个独立候选 activation，仍须按冻结光学输入准入核查。排除清单见 `data/manifests/impactmesh_evaluation_process_exclusions.csv`。

这是一项**训练已启动、但未读取校准/测试标签或任何测试分数时的过程安全修订**，不改已经训练的模型、训练图块、归一化、光学阈值、温度目标或统计口径。原 20/20 设计与修订后的 19/19 有效过程数都须在论文报告，不把它写成最初冻结设计的一部分。不同命名风暴如 Gezani/Fytia 仍视为不同事件。
