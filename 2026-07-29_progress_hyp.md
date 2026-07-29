# Hand-Object 4D Reconstruction Progress

更新时间：2026-07-29

## 1. 当前工作目标

当前工作的目标是构建一套可用于 DexYCB 并能够进一步迁移到 wild 场景的单目/RGB-D 手物 4D 重建流程，统一恢复：

- 手部 MANO mesh 与时序轨迹。
- 物体 mesh、6DoF pose 与时序轨迹。
- 手和物体在相机坐标系下的相对位置。
- 后续可用于接触、穿模和局部手指姿态优化的时序 HOI 初始化。

现阶段采用模块化方案：

```text
RGB / RGB-D sequence
    |
    +-- HandFlow ----------------------> hand mesh initialization
    |
    +-- SAM3D -------------------------> object mesh
    |
    +-- FoundationPose + TAPIR --------> object pose and motion evidence
    |                                      |
    |                                      +-- static/dynamic segmentation
    |                                      +-- EKF/RTS stabilization
    |
    +-- Stage1 hand global refiner ----> hand global placement correction
    |
    +-- Pi3X geometry cache (planned) -> hand-object relative-depth correction
    |
    +-- Stage2 HOI optimization -------> contact and local MANO refinement
```

## 2. 当前整体进展

### 2.1 DexYCB 全流程

已经跑通从 DexYCB 序列准备到 4D 手物可视化的完整流程：

1. 读取 RGB、深度、相机内参和已有 mask。
2. 使用 SAM3D 从候选帧生成物体 mesh。
3. 使用 FoundationPose 估计逐帧物体 pose。
4. 使用 HandFlow 估计逐帧手部 mesh。
5. 将手和物体转换到统一相机坐标系。
6. 使用 Viser 和离线视频渲染检查原始结果、修正结果和 DexYCB GT。

目前已经具备单序列运行、批量缓存、训练监督导出、推理和可视化脚本。

### 2.2 数据与缓存

已基于 FoundationPose QA 通过的数据建立训练清单：

| Split | 序列数 |
| --- | ---: |
| Train | 3592 |
| Val | 204 |
| Test | 673 |

其中：

- Train/Val 的 HandFlow 结果已经批量导出。
- Train/Val 的物体时序滤波结果已经批量导出。
- 训练窗口已经修复尾帧未覆盖问题。
- 已生成 Stage1 所需的手、物体和 GT 监督缓存。

数据主目录：

```text
reconstruction/data/dexycb/hybrid_training_v1
```

## 3. 物体重建与跟踪

### 3.1 SAM3D 物体 mesh

已经可以从指定候选帧和 object mask 生成 SAM3D 物体 mesh，并用于 FoundationPose 跟踪和后续手物可视化。当前 shape bank 和尺度接口已经接入完整流程，后续仅按需要进行候选帧、mask 和 mesh 后处理等工程性改进，不作为现阶段研究主线。

### 3.2 FoundationPose 时序稳定

原始 FoundationPose 在物体运动、遮挡和起停边界处存在明显抖动或滞后。当前已经完成：

- 使用 TAPIR 在 object mask 内提取 2D tracks。
- 使用 PnP/3D motion evidence分析相邻帧物体运动。
- 根据轨迹速度进行静态/动态分段。
- 静态段进行共享 pose 或静态约束。
- 动态段在原始 FoundationPose 上使用 EKF/RTS 平滑。
- 对分段边界增加连续性处理，减少静态段与动态段之间的跳变。

该流程不依赖自行训练的 checkpoint，属于 FoundationPose + 2D tracking + 传统滤波的几何时序后处理。

批量紧凑结果位于：

```text
reconstruction/data/dexycb/hybrid_training_v1/object_motion_filter_v2_compact
```

当前效果：

- 物体高频抖动已经明显降低。
- 动态段整体运动能够保留。
- 少数速度变化较大的边界仍可能受 FoundationPose 原始误差影响。

## 4. 手部全局位置修正

### 4.1 HandFlow 初始化问题

HandFlow 能提供较完整、时序相对稳定的 MANO mesh，但仍存在：

- 相机深度方向漂移。
- 遮挡严重时手部整体位置跳变。
- 个别序列尾部即使 RGB 中手物基本静止，HandFlow 仍会向物体内部漂移。
- 初始手物相对位置可能表现为穿模或虚接。

固定分析序列：

```text
20200813-subject-02__20200813_145612__839512060362
```

该序列在 70--73 帧出现明显 HandFlow 深度漂移，而过滤后的物体 translation 基本稳定。

### 4.2 Stage1 Global Hand Refiner

已经训练 Stage1 全局手部位置修正网络。当前 Stage1：

- 固定过滤后的 FoundationPose 物体 pose。
- 固定 HandFlow 的手指姿态和 MANO shape。
- 只预测手部整体平移，最新版本主要预测沿 camera ray 的有符号深度修正。
- 使用时序 Transformer 聚合窗口内信息。

基础输入包括：

- HandFlow wrist/palm 几何。
- 手部速度和加速度。
- 过滤后的 object center、rotation 和 velocity。
- wrist-object 相对位置和相对速度。
- HandFlow root rotation。
- camera ray 和有效帧标记。

后续加入了 object-local wrist/palm、物体尺度以及手表面到物体表面的统计特征，最新几何版本输入维度为 139。

Stage1 不再承担：

- 接触点优化。
- 穿模优化。
- MANO finger pose 优化。
- object pose 优化。

这些任务计划放到后续 Stage2。

### 4.3 Stage1 定量结果

一个主要全量验证结果为：

```text
Initial wrist median:   22.15 mm
Corrected wrist median: 15.77 mm

Initial wrist p90:      49.94 mm
Corrected wrist p90:    37.90 mm
```

纯 ray-depth 版本在验证集上的结果为：

```text
Initial ray-depth median:   19.14 mm
Corrected ray-depth median: 11.07 mm

Initial ray-depth p90:      48.38 mm
Corrected ray-depth p90:    35.50 mm
```

说明网络在整体统计上能够修正 HandFlow 的相机空间 placement，尤其是深度误差。但当前仍存在两个问题：

1. 对少数困难帧修正不足，尾部漂移仍可能导致穿模。
2. 极端样本的 max error 没有稳定改善，说明仅依靠当前输入难以可靠判断深度修正方向和幅度。

## 5. 已尝试的时序与几何优化

### 5.1 接触/穿模序列优化

已经尝试 CHOIR 风格的候选接触点和无 SDF 穿模约束：

- 从预定义的手部候选接触顶点中选择指尖、指腹和掌心点。
- 在物体 mesh 上寻找最近表面点。
- 根据距离和法线方向建立 contact correspondence。
- 在若干帧内复用并周期性更新接触对应。
- 联合 contact、penetration、2D projection、anchor 和时序项优化手部整体平移。

实验结果表明，仅用不稳定的候选接触点直接推动整只手，容易出现：

- 为了减小 contact distance 而增加其他区域穿模。
- 无接触帧被错误拉向物体。
- 最近表面和局部法线不足以单独提供稳定的穿模修正方向。

因此当前没有将 contact loss 放入 Stage1，而是保留到几何和深度更稳定后的 Stage2。

### 5.2 基于物体轨迹的手部漂移修正

针对固定序列尾部 HandFlow 漂移，尝试利用稳定后的 object trajectory 约束 hand-object relative motion：

- 当物体近似静止且手物交互状态稳定时，抑制手相对物体的异常速度。
- 对 70--71 帧的 HandFlow 大跳变有明显修正效果。
- 证明物体轨迹可以作为识别手部异常漂移的重要时序信号。

但基于人工 carry/hold 区间的规则不能直接泛化到所有序列，因此目前将其作为问题诊断和几何先验，而不是最终主线方案。

## 6. Pi3X 几何 Head 进展

### 6.1 当前动机

仅靠 HandFlow 与物体轨迹无法可靠确定手物相对深度，尤其在遮挡严重时。下一步计划使用 Pi3X geometry head cache 提供场景几何观测：

- 用物体可见区域将 Pi3X 几何与过滤后的 object mesh/pose 对齐。
- 在同一相机坐标系中提取可见 hand-region geometry。
- 估计 HandFlow mesh 与可见手部几何之间的相对深度残差。
- 对遮挡帧使用前后可靠帧和时序模型传播，而不是使用人工 hold 区间。

### 6.2 已完成

- 已实现 Pi3X geometry head cache 的单帧导出接口。
- 已显式输入 DexYCB GT camera intrinsics。
- 已实现基于物体 mask 和 object mesh 的几何对齐接口。
- 已实现 hand mask 区域的深度观测提取。
- 已实现 Pi3X point cloud、object mesh 和 HandFlow mesh 的 Viser 对齐检查。
- 已加入完整点云、object points、hand points 和 hand mesh 的 PLY/OBJ 导出。

### 6.3 当前发现

当前 Pi3X cache 仍属于独立实验结果，尚未接入 Stage1 或主流程。现阶段需要重点验证其坐标、尺度和相对深度可靠性：

1. 输入真实相机内参后，Pi3X 点云是否处于正确相机坐标系。
2. object-mask point cloud 是否与过滤后的 object mesh 可见面正确对齐。
3. hand-mask point cloud 是否能在可见区域提供可靠的相对深度。
4. 遮挡帧是否能够由前后可靠观测和时序特征补全。

## 7. Highlights

1. 已跑通 DexYCB 上 HandFlow + SAM3D + FoundationPose 的完整 4D 手物重建流程。
2. 已建立 3592/204/673 条 train/val/test 序列清单及对应缓存体系。
3. 已完成基于 TAPIR 静动态分段和 EKF/RTS 的物体时序稳定流程，并完成 Train/Val 批量导出。
4. 已定位物体高频抖动与 HandFlow 相对深度漂移是两个不同问题，并分别处理。
5. 已训练 Stage1 手部全局 placement 网络，将验证集 wrist median error 从约 22.2 mm 降至约 15.8 mm。
6. 已修复时序窗口尾帧未覆盖问题，确保所有有效尾帧参与训练和推理。
7. 已建立 GT hand、GT object、原始结果和修正结果的 Viser/离线多视角对比工具。
8. 已验证稳定物体轨迹能够识别并抑制部分 HandFlow 异常漂移，为可学习的 object-guided temporal model 提供了依据。
9. 已完成 Pi3X + GT camera intrinsics 的单帧 geometry cache 接口，为后续相对深度修正提供基础。

## 8. 当前瓶颈

1. FoundationPose 虽已平滑，但原始跟踪在快速起停和严重遮挡处仍可能有系统误差。
2. HandFlow 在遮挡和序列尾部可能发生整体深度漂移。
3. Stage1 的平均指标已改善，但困难帧仍缺少足够可靠的几何证据。
4. Pi3X 的真实内参版本尚需完成点云与物体 mesh 的定量对齐验证。
5. 当前还没有可靠的、可泛化的接触状态和接触顶点预测。
6. 最近表面距离和局部法线不足以单独提供稳定的穿模修正方向。

## 9. TODO

### 9.1 P0：Pi3X 相对深度验证

- 完成 Pi3X geometry head cache 的单帧与短序列导出。
- 确认 GT intrinsics 的 resize/crop 变换正确。
- 可视化完整 Pi3X 点云、object-mask 点云、过滤后的 object mesh、hand-mask 点云和 HandFlow mesh。
- 以 RGB-D 深度作为实验期参考，定量评估 object alignment 和 hand relative-depth observation。
- 将绝对深度改为 object-anchored relative depth，减少窗口 gauge 对结果的影响。

### 9.2 P1：基于 Pi3X 几何的手物位置修正

- 从每帧 Pi3X cache 提取：
  - object-visible geometry。
  - hand-visible geometry。
  - confidence、可见率和深度分布。
- 仅在可靠可见帧生成 hand-object relative-depth observation。
- 对遮挡帧使用双向时序传播和 uncertainty weighting。
- 将 Pi3X observation 作为 Stage1 的额外输入/监督，而不是直接替换 HandFlow。
- 同时使用过滤后的 object trajectory 约束相对速度和相对加速度。

### 9.3 P2：Stage2 手物交互优化

在 Stage1 placement 和 Pi3X 相对深度稳定后：

- 固定或小范围开放 object pose。
- 小范围优化 hand translation 和 root rotation。
- 开放 MANO local pose，保持 shape 固定。
- 使用候选接触顶点和时序稳定的 contact correspondence。
- 加入 penetration、contact、2D joint reprojection、MANO prior 和 temporal loss。
- 避免逐帧独立更新接触点，改为交互片段级接触状态和稀疏更新。

## 10. 下一步执行顺序

1. 先验证 Pi3X geometry head cache 与 object mesh 的相机坐标对齐。
2. 再验证 Pi3X hand-visible geometry 是否能正确检测已知 HandFlow 深度漂移。
3. 若相对深度可靠，将 Pi3X geometry feature 接入 Stage1，训练 object-anchored relative-depth refiner。
4. 最后开展 Stage2 手物交互优化，处理穿模、虚接和局部 MANO pose。

## 11. 可交付物状态

| 模块 | 状态 | 备注 |
| --- | --- | --- |
| DexYCB 数据准备与 manifest | 已完成 | Train/Val/Test 已建立 |
| HandFlow 批量导出 | 已完成 | Train/Val 已缓存 |
| SAM3D object mesh pipeline | 已跑通 | 已接入 shape bank 和后续流程 |
| FoundationPose object tracking | 已跑通 | 原始结果存在抖动/滞后 |
| TAPIR 静动态分段 | 已完成 | 已用于批量物体处理 |
| Object EKF/RTS stabilization | 已完成 | Train/Val 紧凑结果已导出 |
| Stage1 global hand refiner | 已完成第一版 | 平均指标改善，困难帧仍需几何输入 |
| Contact/penetration sequence optimization | 原型完成 | 暂不并入 Stage1 |
| Object-guided hand drift correction | 原型完成 | 有效但规则版本不可直接泛化 |
| Pi3X + GT intrinsics cache | 进行中 | 单帧接口已完成，待系统验证 |
| Stage2 HOI/contact optimization | 待开展 | 依赖稳定 placement 和几何 |

## 12. 阶段性结论

目前已经从“单独恢复手或物体”推进到“统一相机坐标系下的时序手物重建与误差诊断”。物体侧的时序抖动已经通过 tracking evidence 和传统滤波得到明显改善；手侧的全局 placement 网络在整体指标上有效，但遮挡场景中的相对深度仍是主要瓶颈。

下一阶段不再继续单纯增加手部回归网络容量，而是优先使用带真实相机内参的 Pi3X geometry head cache，建立以稳定物体为锚点的手物相对深度观测。在相对 placement 稳定后，再开展接触、穿模和局部 MANO pose 联合优化。
