# V15 Scaling 数据集简表

更新时间：2026-08-27

## 统计口径

- `Ego 图像`只统计第一视角相机图像。
- `总图像`会把同一时刻的不同相机视角分别计数，不能直接当作时序帧数。
- 没有官方统一统计的项目标为“未单独公布”，不把估算值写成确定数字。
- V15 当前需要的是 RGB、相机参数、2D/3D hand joints 或 MANO，以及连续帧。RGB-D 并非必需。

## Ego 与轨迹主数据集

| 数据集 | Hand GT | Object / HOI GT | Ego 图像 | 总图像 | 单个序列大概多长 | 对 V15 的价值 |
| --- | --- | --- | ---: | ---: | --- | --- |
| **DexYCB** | 单手 2D/3D joints、valid、MANO、mask | YCB mesh、每帧 6D pose | 0 | 约 582K | 1,000 序列；约 72 个同步帧、2.4 秒；8 views | 基准和监督链路验证，规模偏小 |
| **H2O** | 双手 21 joints、valid、MANO | object pose、action、hand-object interaction | 约 114K | 约 571K | 通常数百帧、约 5-10 秒；1 ego + 4 exo | 双手 ego、真实遮挡，优先级高 |
| **HOI4D** | MANO、21 个 2D joints、hand motion mask | object pose/segmentation、action、4D HOI | 约 2.4M | 约 2.4M | 4,000 序列；平均约 600 帧、20 秒 | 大规模 ego 轨迹主力 |
| **AssemblyHands** | 双手 3D joints、valid、bbox、标定 | assembly action，物体精确 6D pose 较弱 | 约 490K | 约 3.03M | 约 82 条长序列；平均约 1.5K 同步帧、50 秒 | 长时双手轨迹与遮挡训练 |
| **HOT3D** | 双手 MANO/UmeTrack skeleton、shape/pose、crop | object models、6D pose、分割 | 约 3.7M 头戴相机图像 | 约 3.7M | recording 约 2 分钟、3,600 同步帧；clips 为 150 帧 | 长时 ego、多手、多物体，优先级最高 |
| **ARCTIC** | 双手 MANO、3D joints/mesh、global translation | articulated object mesh/pose、contact | 约 237K | 约 2.1M | 339 序列；平均 698 同步帧、23.3 秒；1 ego + 8 exo | 精确双手和接触监督 |
| **TACO** | 双手 3D hand pose/mesh | tool 与 target object mesh/pose、segmentation、contact/interaction | 有 ego；官方未单列数量 | 约 5.2M | 约 2.5K motion sequences；混合多视角，不能直接用图像数推算时长 | 工具使用与双物体交互，优先级高 |
| **Re:InterHand** | 双手 3D joints/mesh、MANO、mask、相机参数 | 无真实 object pose 主标注 | 每种 ego 设置约 148K | 约 1.53M | 10 个长 capture；ego 平均约 14.8K 帧、8.2 分钟 | 双手长序列和遮挡辅助 |

## 手物精标与大规模多视角数据集

| 数据集 | Hand GT | Object / HOI GT | Ego 图像 | 总规模 | 单个序列大概多长 | 对 V15 的价值 |
| --- | --- | --- | ---: | ---: | --- | --- |
| **OakInk2** | 双手 MANO pose/shape/translation、SMPL-X | object mesh、每帧 SE(3)、articulated object、affordance、task labels | 官方 release 为 4-camera multi-view，未单列 ego | 627 sequences、4.01M images、75 objects | 平均约 1.6K 同步时刻；约 53 秒（按 30 fps 估算） | 长时双手手物轨迹，优先级最高 |
| **ContactPose** | 3D hand pose，可拟合 MANO | object pose/mesh、dense contact map | 0 | 超过 2.9M RGB-D images、2,306 unique grasps、25 objects | 以静态/短抓取观测为主，不是连续动作轨迹 | 接触与抓取几何辅助；不适合作为轨迹主数据 |
| **OakInk** | MANO、21 joints、778 vertices | object mesh、每帧 SE(3)；OakInk-Shape 含大量 CAD/grasp | 0；4-camera multi-view | 当前未过滤版约 314K frames；100 real objects、1,801 object models | 官方未给统一序列长度 | 精确 hand-object pose 与物体泛化 |
| **HO3D v3** | 右手 MANO、2D/3D joints | object mesh、6D pose | 0 | 103,462 annotated RGB-D images | 约 77 sequences，通常为短交互片段 | 经典高质量手物监督，规模中等 |
| **GRAB** | MANO/SMPL-X 全身与双手 | object mesh/pose、详细 contact | 0；无真实 RGB 主数据 | 1,334 MoCap sequences、约 1.62M frames、51 objects | 动作长度不一，属于连续 MoCap | 接触、运动先验和合成训练，不贡献真实外观 |
| **BEHAVE** | SMPL/SMPL-H 类全身，手部细节弱于 MANO 数据集 | object mesh、6D pose、contact、mask | 0 | 321 sequences、8 subjects、20 objects、4 RGB-D views | 官方未公布统一平均帧数 | 全身 HOI 辅助，不应作为手深度主数据 |
| **InterCap** | SMPL-X | object mesh、6D pose、contact-assisted pseudo GT | 0 | 223 videos、67,357 同步帧、404,142 RGB-D images | 平均约 302 同步帧、10 秒；6 views | 全身 HOI 与物体轨迹辅助 |
| **GigaHands** | 双手 MANO、2D/3D joints、shape/pose | object 6DoF、mesh、segmentation、文本动作标注 | 0；大规模 multi-view | 14K clips、183M images、34 hours、417 objects | 平均约 8.7 秒、约 262 同步帧；大量相机视角 | 极大规模外观和双手运动；建议抽视角使用 |
| **InterHand2.6M** | 单手/双手 42 joints、valid、MANO、bbox、相机参数 | 无 object pose 主标注 | 0 | 5 fps 约 2.59M；30 fps 约 12.47M | 官方未给统一平均长度 | 双手几何预训练，不解决 hand-object relation |

## TB 级数据组合

### Ego / 头戴相机数据

**H2O + HOI4D + HOT3D + ARCTIC + TACO + Re:InterHand**

- H2O、ARCTIC 同时包含 ego 和外部多视角相机。
- HOT3D 来自 Aria/Quest 头戴设备，包含同一时刻的多路头戴相机图像。
- TACO 同时包含 egocentric 和 third-person views，官方未单独公布 ego 图像数量。
- Re:InterHand 的 ego 数据是渲染得到的双手序列，不是真实头戴相机采集。

### 多视角 hand-object 数据

**DexYCB + OakInk + OakInk2 + HO3D + ContactPose + GigaHands**

- DexYCB、OakInk、OakInk2 和 HO3D 提供 hand pose 与 object pose，可直接构造相机坐标系下的手物监督。
- ContactPose 主要是静态/短抓取与 dense contact，而不是长时间轨迹。
- GigaHands 的 183M 图像包含大量同步相机视角；图像数量远大于独立时序帧数。

### 多视角双手 / 全身交互数据

**AssemblyHands + InterHand2.6M + BEHAVE + InterCap**

- AssemblyHands、InterHand2.6M 主要补充双手 pose、遮挡和跨视角几何。
- BEHAVE、InterCap 主要补充全身 HOI 与 object pose，手部精度弱于 MANO 手部专用数据集。

### MoCap 与几何先验

**GRAB**

GRAB 不以真实 RGB 为主，适合作为 hand motion、contact 和 object-relative geometry 的辅助监督。

### 约 5 个数据集的混合主集合

**HOI4D + HOT3D + TACO + OakInk2 + GigaHands**

| 采集类型 | 数据集 | 作用 |
| --- | --- | --- |
| Ego | **HOI4D** | 约 2.4M 第一视角帧，提供大规模连续 hand-object interaction |
| Ego / 头戴多相机 | **HOT3D** | 约 3.7M 图像，覆盖长时、多手、多物体和真实遮挡 |
| Ego + third-person | **TACO** | 约 5.2M 图像，覆盖双手、tool-object 与双物体交互 |
| 多视角 | **OakInk2** | 约 4.01M 图像，提供双手 MANO、object SE(3)、articulated object 和任务标签 |
| 大规模多视角 | **GigaHands** | 约 183M 图像，覆盖大量双手动作、物体和视角；训练时应抽取部分相机视角 |

该组合以规模为第一筛选条件，同时包含真实 ego 和多视角数据。HOI4D、HOT3D、TACO 构成 ego/混合视角部分，OakInk2、GigaHands 构成多视角部分；TACO 同时连接两类采集设置。

DexYCB、H2O、ARCTIC、OakInk 和 HO3D 可继续作为精标验证集或小比例监督数据，但不计入这 5 个大规模主数据集。若保存完整时间采样的 RGB/视频、相机参数及标注，而不保存暂时不用的 depth 和重复中间 cache，整体会达到 TB 级；实际占用取决于视频编码、采样帧率和保留的相机数量。

## 官方资料

- [DexYCB](https://dex-ycb.github.io/)
- [H2O](https://github.com/taeinkwon/h2odataset)
- [HOI4D](https://hoi4d.github.io/)
- [AssemblyHands](https://assemblyhands.github.io/)
- [HOT3D](https://facebookresearch.github.io/hot3d/)
- [ARCTIC](https://arctic.is.tue.mpg.de/)
- [TACO](https://taco2024.github.io/)
- [OakInk2](https://oakink.net/v2/)
- [OakInk](https://oakink.net/)
- [ContactPose](https://contactpose.cc.gatech.edu/)
- [HO3D](https://github.com/shreyashampali/ho3d)
- [GRAB](https://grab.is.tue.mpg.de/)
- [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/)
- [InterCap](https://intercap.is.tue.mpg.de/)
- [GigaHands](https://github.com/brown-ivl/GigaHands)
- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)
- [Re:InterHand](https://mks0601.github.io/ReInterHand/)
