# Hand Depth Refinement 实验复盘

日期：2026-08-11

## 1. 目标

当前任务是利用 Pi3X 的时序几何信息修正 HandFlow 的三维手部结果，重点改善 wrist/hand 在相机射线方向上的深度偏差，同时满足：

- 不依赖测试时 GT；
- 不让错误的物体姿态直接带动手部；
- 保持 HandFlow 已经较好的二维投影、手型和局部姿态；
- 能修正少量漂移帧，又不破坏本来正确的帧；
- 左右手使用一致的物理定义和评测方式。

最终将问题收敛为：保持 HandFlow wrist 的相机射线不变，只预测或优化射线方向上的一维深度。

## 2. 数据和评测链路

### 2.1 监督数据

训练监督来自 DexYCB GT wrist/root pose，HandFlow 提供初始手部结果，FoundationPose/SAM3D 提供物体姿态，Pi3X cache 提供时序特征、点图和 metric token。

训练导出审计结果：

- supervision streams：3438；
- window streams：2939；
- train windows：40220；
- duplicate windows：0；
- translation/rotation round-trip 数值误差接近零；
- train/val window manifest 修复后，train 2939 streams、val 152 streams，重叠为 0。

早期曾误用包含训练流的 val windows，出现 656 个 train/val 重叠 stream；该问题已经修复，之后结果均使用独立 val stream。

### 2.2 统一评测指标

- `initial/corrected ray depth`：相机射线方向的绝对误差；
- `initial/corrected translation`：完整三维 wrist translation error；
- `degraded_fraction`：修正后误差大于修正前误差的帧比例；
- `worse > 1/2/5 mm`：比单纯 degraded fraction 更有意义的退化幅度；
- left/right 分组；
- 初始 ray error 分桶：`0-5`、`5-15`、`15-30`、`30+ mm`；
- feature zero、time reverse、spatial shuffle 等消融，检查网络是否真正使用 Pi3X。

需要注意：`degraded_fraction` 把任意微小恶化都计为退化，因此必须同时报告超过 1/2/5 mm 的比例。

## 3. 前期坐标系和可视化问题

### 3.1 Object-frame supervision

最初将 wrist 变换到物体坐标系，预测 hand SE(3) 的绝对值或 residual。数学上的坐标变换 round-trip 正确，但该路线强依赖 FoundationPose 和 SAM-to-YCB canonical alignment。

已排查的问题包括：

- SAM3D canonical mesh 与 YCB canonical mesh 的轴和尺度映射；
- FoundationPose 与 GT object pose 的旋转/平移误差；
- 左手输入镜像与 physical camera frame 的恢复；
- 使用 mesh center 代替 wrist joint 导致的可视化错位；
- 对 normalized-left pose 重复镜像；
- object-frame prediction 写回 camera mesh 时的行/列向量变换约定。

在代表性左手序列中，relative target 重新计算误差约为 `0.0001 mm`，证明导出的 relative target 与实现公式一致；但 target camera wrist 与 GT wrist 仍可相差约 26.5 mm，主要来自 object pose/canonical 路线本身。这个结果说明 object-frame 监督虽然代数正确，却给手部修正引入了不必要的物体姿态误差。

### 3.2 结论

手部和物体应分开修正：

- 手部直接在 camera frame 对 GT hand 学习；
- 物体姿态由独立分支处理；
- 物体信息最多作为弱上下文，不能直接决定手部全局 pose。

## 4. 各版本实验

### 4.1 Object-frame absolute pose 系列

#### 基础 MLP + BiGRU

输入包含显式 hand/object pose 相关量，共 42 维，预测 object-frame hand translation 和 rotation。

代表性 val 结果：

- translation：`39.27 -> 57.33 mm`；
- rotation：`30.53 -> 75.84 deg`。

模型能降低训练 loss，但跨流泛化明显差于初始 HandFlow。

#### Object embedding

加入 17 类 object embedding 后：

- translation：`39.27 -> 53.94 mm`；
- rotation：`30.53 -> 80.68 deg`。

object identity 没有解决几何对应问题，反而更容易学习类别先验。

#### Pi3X relative cross-attention

使用 Pi3X feature、metadata 和 hand/object relative cross-attention。早期小样本训练结果为：

- translation：`39.27 -> 65.50 mm`；
- rotation：`30.53 -> 83.74 deg`。

在扩大和平衡训练后，候选结果改善到：

- translation：`39.27 -> 37.57 mm`；
- rotation：`30.53 -> 27.27 deg`。

再加 initial/candidate selector 后可到：

- selected translation：36.94 mm；
- selected rotation：20.10 deg。

但该路线仍然显式依赖 initial/object pose，而且左右手和物体坐标可视化链路复杂，最终没有作为主线。

### 4.2 Object-frame local residual

`object_frame_hand_local_pi3x_se3_residual_v2` 不把 initial SE(3) 编码成普通网络输入，只用于 token localization 和输出 composition。

全量结果：

- translation：`39.27 -> 34.04 mm`；
- rotation：`30.53 -> 11.54 deg`。

rotation 收益明显，但 object-frame target 和可视化仍受 object pose、左手镜像与 canonical mapping 影响。该版本证明 residual composition 可行，但没有解决“手是否应该跟随错误 object pose”的根本问题。

### 4.3 V8：camera-frame residual 与显式输入

V8 已经在 camera frame 预测 HandFlow hand pose residual，方向上比 object-frame 直接回归更合理；但网络仍编码 HandFlow initial state、object rotation 等显式量，存在依赖初始 pose 或 object pose 走捷径的风险。后续 V9 的主要目的不是重新证明 residual 有效，而是去掉这些显式 pose 输入，只保留视觉/几何 observation，并直接使用 camera-frame GT hand 监督。

V8 与后续版本的监督、窗口和 unique-frame 审计口径不完全一致，因此本文不把其旧指标与 V9-V12 直接横向比较。

### 4.4 V9：camera-frame observation-only residual

`v9_camera_hand_residual_observation_only_v1` 去掉显式 object pose 输入，直接对 camera-frame GT wrist/root 学习 HandFlow residual。

val：

- translation：`21.30 -> 16.96 mm`；
- rotation：`5.96 -> 5.43 deg`。

这是第一次在不依赖 object pose 的情况下稳定改善 translation，说明 camera-frame 监督方向正确。rotation 初始已经较好，因此后续重点转为 ray depth。

### 4.5 V9.1：camera-ray depth residual

`v9_1_camera_ray_depth_residual_observation_only_v2` 只预测 wrist 沿 HandFlow 相机射线的一维 residual，保持二维投影不变。

独立 val windows：

- ray depth：`18.65 -> 14.86 mm`；
- translation：`21.22 -> 18.34 mm`；
- degraded fraction：约 40.5%。

按 unique stream/frame 审计：

- all ray：`19.21 -> 15.24 mm`；
- left：`21.12 -> 19.30 mm`；
- right：`18.06 -> 12.69 mm`。

该版本确认：主要可修正量确实位于 camera ray 方向，但 left 分支收益持续弱于 right。

### 4.6 V9.2：Pi3X feature trajectory

`v9_2_pi3x_feature_trajectory_ray_depth_v1` 使用 Pi3X hand/object/context tokens 和时序网络预测 ray-depth residual。

代表性 val：

- ray：`18.65 -> 14.93 mm`；
- translation：`21.22 -> 18.45 mm`；
- left ray：`21.12 -> 18.93 mm`；
- right ray：`17.14 -> 12.33 mm`；
- degraded fraction：39.7%。

消融：

- normal：14.93 mm；
- hand feature zero：16.14 mm；
- object feature zero：16.24 mm；
- all feature zero：17.80 mm；
- feature time mean：15.10 mm。

该结果证明 Pi3X feature 在这个版本中确实被使用，并且包含有益信息。但对初始误差小于 5 mm 的帧，模型明显容易过修：该组 degraded fraction 约 70%。

加入 small-anchor/degradation 约束后，总 degraded fraction 可下降到约 37%，但 left 仍弱，且小误差帧仍是主要风险。

### 4.7 V9.3/V9.4：joint-conditioned 与 no-op head

目标是让网络判断“当前帧是否根本不需要修正”，并将 Pi3X feature 与 HandFlow joints 建立空间对应。

#### V9.3 projected joint tokens

- ray：`18.65 -> 15.28 mm`；
- degraded fraction：41.7%；
- val no-op AUC：0.613；
- balanced accuracy：0.574。

分类信号存在，但不足以稳定区分 2 mm 与 15 mm 误差。

#### V9.4 dense joint sampling

单序列 overfit：

- ray：`9.33 -> 7.08 mm`；
- no-op AUC：0.914。

但 50-train/20-val stream pilot：

- ray：`16.81 -> 15.37 mm`；
- no-op AUC：0.530；
- dense feature zero 反而略优；
- time reverse 基本不影响结果。

结论：单序列 overfit 证明结构有容量，但跨序列时 dense joint features 没有形成稳定可泛化的几何判据，模型主要学习数据先验。

### 4.8 V10：Pi3X hand-neighborhood depth

该版本围绕 projected hand joints，从 Pi3X point/features 中采样局部 neighborhood，尝试建立更精确的几何对应。训练可以正常收敛，但跨序列收益仍弱，未解决 left/right 差异，也没有得到足够强的 feature ablation 差距。

结论：仅提高空间采样精度不能解决 Pi3X 非 metric depth 与真实 wrist depth 之间的标定问题。

### 4.9 V11：ret_point/ret_metric 与绝对深度

#### V11 absolute depth

`v11_pi3x_metric_absolute_ray_depth_v1` 不输入 HandFlow 显式深度，仅根据 Pi3X ret_point/ret_metric tokens 预测绝对 wrist depth。

单序列 overfit 到 epoch 108 后：

- ray：`11.47 -> 0.77 mm`；
- translation：`18.35 -> 5.52 mm`；
- degraded fraction：4.2%。

消融中 point zero、all zero、time reverse 都会导致严重退化，说明单序列上模型确实使用了 Pi3X 时空特征。

但 50/20 stream pilot：

- ray：`16.81 -> 39.07 mm`；
- translation：`19.41 -> 40.55 mm`；
- degraded fraction：73.0%。

绝对深度映射可以被记忆，但无法跨 sequence/camera/object 稳定泛化。

#### V11.1 metric-point residual

改回 bounded residual，并联合 decoder features、points 和 metric features：

- ray：`16.81 -> 14.68 mm`；
- translation：`19.41 -> 18.83 mm`；
- degraded fraction：37.5%。

但消融结果几乎不变：

- normal：15.20 mm；
- decoder zero：15.29 mm；
- point zero：15.23 mm；
- metric zero：15.27 mm；
- all Pi3X zero：15.29 mm。

说明该 residual 网络虽然指标略有改善，但主要依赖训练分布偏置，没有真正依赖 Pi3X。

#### V11.2 HandFlow latent + Pi3X

加入 HandFlow pre-translation latent，使网络能够将 HandFlow 当前状态与 Pi3X 观测联系起来。latent replay 已验证能够完全重建原 HandFlow mesh，排除了重新推理引入的状态漂移。

但 left/right target 分布存在明显差异：

- pilot val left target ray median：约 1.33 mm；
- pilot val right target ray median：约 16.09 mm；
- 扩展样本后 train left/right median 约为 6.34/11.17 mm。

因此共享网络很容易学习整体正向修正先验，对 left 产生系统性过修。

#### V11.3 canonical-right

将左手输入统一镜像到右手 canonical representation：

- normal all ray：14.93 mm；
- left：18.06 mm；
- right：12.68 mm；
- latent/Pi3X zero 与 normal 差异很小。

left-only 训练结果为 `15.98 -> 16.00 mm`，后续 epoch 持续过拟合并恶化。说明左右 canonical 化本身正确，但不是 left 分支异常的主要原因；主要问题仍是 target distribution 和几何信号可观测性。

#### V11.4 per-joint rigid

增加 per-joint depth observation、rigid consistency、joint reliability 和 no-op head。left-only 结果：

- ray：`15.98 -> 16.37 mm`；
- translation：`20.58 -> 21.28 mm`；
- degraded fraction：53.1%。

no-op head 塌缩为全 negative，joint reliability 塌缩为全 positive。增加辅助 head 没有创造新的可比较几何信号。

### 4.10 V12：metric-preserving absolute depth

`v12_metric_preserving_absolute_hand_depth_v1` 尝试保留 Pi3X metric token 的绝对尺度信息，并融合 HandFlow latent、joint samples 和时序信息。

pilot val：

- ray：`17.03 -> 21.54 mm`；
- translation：`20.06 -> 24.50 mm`；
- degraded fraction：62.2%；
- left ray：`16.22 -> 25.11 mm`；
- right ray：`17.86 -> 18.39 mm`。

消融证明模型确实使用 decoder、metric 和 HandFlow latent，但使用这些信号后仍朝错误方向泛化。问题不是网络忽略 Pi3X，而是 Pi3X metric 与真实 wrist metric depth 之间没有稳定的跨流映射。

### 4.11 V12.1：Pi3X geometry anchor 可观测性审计

直接检查 Pi3X point cloud 聚合深度能否作为 wrist 几何锚点。

结果：

- raw center/median/confidence-mean anchor error 为数百毫米；
- 使用 train affine calibration 后仍约 70-90 mm；
- 与真实 correction 的相关性通常只有 `0.02-0.08`，部分 left 组甚至为负。

这说明当前 Pi3X point/metric 输出在序列内具有协调性，但不能直接作为跨序列的 metric wrist depth 测量。继续堆叠网络层、增加 token 数量或扩大 receptive field，不能从根本上修复不可比较的几何量。

### 4.12 Wrist Depth Optimization 时序后处理

参考相关工作，只优化每帧 wrist 沿相机射线的深度：

\[
E(d)=\sum_t w_t(d_t-d_t^{HF})^2+
\lambda\sum_t\|X_{t-1}(d_{t-1})-2X_t(d_t)+X_{t+1}(d_{t+1})\|^2.
\]

在测试序列上，`lambda=0.2`：

- wrist acceleration median：`4.02 -> 3.13 mm/frame^2`；
- acceleration p90：`11.89 -> 8.07 mm/frame^2`；
- acceleration max：`24.96 -> 11.57 mm/frame^2`；
- ray：`9.65 -> 9.42 mm`；
- translation：`11.76 -> 11.74 mm`；
- translation p90：`26.49 -> 25.76 mm`；
- median/max correction：0.44/5.77 mm。

逐帧审计：

- degraded fraction：50%，但多数是亚毫米变化；
- worse > 1 mm：6.8%；
- worse > 2 mm：2.7%；
- worse > 5 mm：0%。

该方法能明显压制轨迹尖峰，但 acceleration 无法区分错误漂移和真实快速运动。HandFlow `hamer_confidence` 范围只有 0.767-0.886，并且与 bbox confidence 完全一致；加入 confidence weighting 后几乎没有额外收益。

结论：时序优化适合作为保守去抖后处理，但不能代替几何信息修正系统性深度偏差。

## 5. 已确认的关键问题

### 5.1 不是单纯的网络容量问题

多个版本能够在单序列上 overfit，证明模型容量和反向传播链路没有问题。失败主要发生在跨 sequence/object/side 泛化。

### 5.2 Pi3X feature 有信息，但当前没有稳定 metric 对应

V9.2 的 feature-zero 消融和 V11 单序列 absolute-depth 消融证明 Pi3X feature 包含时空信息。问题是网络需要从高维 token 中同时学习：

1. 哪些 token 对应当前 HandFlow 手；
2. Pi3X 内部深度尺度如何映射到相机 metric depth；
3. 当前帧是否需要修正；
4. 应向前还是向后修正；
5. 左右手和不同 stream 的分布差异。

在当前 pilot 数据规模下，这些因素没有被解耦，模型容易学习全局 correction prior。

### 5.3 no-op/gate 不是独立解决方案

无论叫 no-op head、selector 还是 reliability head，如果输入中没有可泛化的误差观测，它只能学习类别比例。单序列 AUC 很高、跨序列接近随机，已经验证了这一点。

### 5.4 Left 分支问题不只是镜像

已验证 left normalized/canonical-right 几何变换正确。left 异常主要来自：

- left/right target ray correction 分布不同；
- pilot stream 数量有限；
- 共享模型学习了偏向 right 的正向修正先验；
- Pi3X/HandFlow 特征对误差方向的可观测性不足。

### 5.5 Object pose 不应直接驱动 hand correction

FoundationPose 即使能重投影到 RGB，也可能在 canonical rotation、对称性和 metric translation 上与 GT 不一致。让手跟随 object pose 会把物体误差注入手部。因此当前主线应坚持“手修手、物体修物体”。

## 6. 当前阶段结论

目前最可靠的结论是：

1. camera-frame、ray-constrained residual 是正确的问题定义；
2. V9/V9.1 能稳定改善 HandFlow，适合作为 baseline；
3. V9.2 证明 Pi3X feature 有用，但小误差帧过修和 left 泛化仍未解决；
4. ret_metric/ret_point 不能直接当作真实 metric wrist depth；
5. absolute-depth 网络单序列可拟合，但跨序列失败；
6. detector confidence 不能有效识别 wrist depth 漂移；
7. Wrist Depth Optimization 能去抖，但对 median metric accuracy 提升有限；
8. 当前瓶颈是缺少一个跨序列可比较、与 HandFlow 当前状态对齐的几何误差信号，而不是缺少更复杂的网络层。

## 7. 建议下一步

### 7.1 短期可交付方案

采用两阶段保守方案：

1. 使用 V9.1/V9.2 bounded camera-ray residual，限制 correction 在合理范围；
2. 使用 `lambda=0.2` 的 wrist temporal optimization 去除个别轨迹尖峰；
3. 对初始 ray error 小的样本加强 no-change loss，但不依赖单独 gate；
4. 报告 unique-frame、left/right、error-bin 和 thresholded degradation 指标。

这条路线不承诺解决所有绝对深度偏差，但可以稳定改善总体 ray error并控制可见退化。

### 7.2 Pi3X 后续研究方向

若继续使用 Pi3X cache，应先构造可比较测量，再训练最终 refiner：

- 将 HandFlow joints 投影到 Pi3X feature/point grid，保留明确对应关系；
- 在每个 stream/window 内做尺度和偏置归一化，不直接跨 stream 比较 raw metric；
- 预测 bounded residual，而不是直接预测 absolute metric depth；
- 将 Pi3X prediction 作为弱 proposal：

\[
E(d)=E_{\mathrm{HF-anchor}}+\lambda_{Pi3X}E_{\mathrm{bounded-proposal}}
+\lambda_{temporal}E_{acceleration}.
\]

- 只有在跨 stream 消融中 normal 明显优于 feature-zero/time-reverse，才进入全量训练。

### 7.3 暂时停止的方向

- 继续增加 object embedding 或显式 object pose；
- 直接使用 raw Pi3X metric/point depth 作为 wrist absolute depth；
- 在没有可观测信号时继续叠加 no-op/reliability head；
- 仅凭单序列 overfit 决定进入全量训练；
- 继续扩大网络而不先验证 feature observability。

## 8. 一句话汇报版本

前期已经排除了坐标系、左手镜像、监督 round-trip 和 train/val 泄漏问题；camera-ray residual 能稳定改善 HandFlow，Pi3X feature 也被证明包含有效信息，但当前 Pi3X metric/point depth 与真实 wrist metric depth 缺少跨序列稳定标定，导致复杂模型主要学习数据先验，尤其在 left 和小误差帧上过修。现阶段最稳妥的方案是 bounded camera-ray residual 加时序 wrist-depth optimization，Pi3X 分支则需要先解决可比较几何测量，再继续全量训练。
