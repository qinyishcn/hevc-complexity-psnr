# 特征字典与可获取时机

所有预测变量由 prediction_features.py 的显式白名单构建；输入表保留 PSNR/MSE 作为验证标签，但它们从不进入模型。

| 参数 | 来源与定义 | 可获取时机与代价 |
| --- | --- | --- |
| target_kbps | 应用配置的目标码率，模型取 log2 | 编码前，直接读取配置 |
| si_2023 / ti_2023 | 原实验 P.910 2023 亮度预处理后的 SI/TI；模型取 log1p | 当前原始帧及前一原始帧已到达，需要像素计算；首帧 TI 缺失，在训练折中填补 |
| pict_type | 最终 I/P/B 类型，逐帧模型使用 I、P 两个指示变量 | 帧类型决定后，编码器或码流可提供；与 CU 模式不同 |
| gop_position | frame % 60 / 59 | 当前固定 GOP 配置与帧计数；更换 GOP 后需要重新建模 |
| qp | x265 CSV 的 QP，编码源码对应 m_avgQpAq，保留日志两位小数精度；不是第一遍 QP，也不是简单的切片头 QP | 当前帧编码完成后，由编码器统计导出；不需要计算源图像与重建图像误差 |
| intra_cu_pct | CSV 首组全帧 CU 模式中，各尺寸 DC/Planar/Angular 加 4x4（8×8 CU 的 NxN 帧内分区）计数比例 | 当前帧模式选择完成后，由编码器 CSV 或统计 API 导出；普通 ffprobe 不直接提供 |
| inter_cu_pct | 普通 Inter + Skip + Merge 的全帧 CU 计数比例；与 intra 互补 | 同上；保存供分析，模型使用 intra 和 skip_merge 两个独立比例，避免再加完全互补的冗余列 |
| inter_nonskip_cu_pct | CSV 中普通 Inter 项合计 | 同上；保存但不额外输入模型 |
| skip_merge_cu_pct | CSV 首组 Skip+Merge 项合计；源码在 isSkipped 分支按 mergeFlag 分开计数，因此合并为跳过类，不将 CSV Merge 误称为所有非跳过 merge 预测块 | 同上；属于帧间预测，不与 inter 总比例并列当互斥类 |
| packet_bytes / packet_bpp | 原始 ffprobe 帧包字节数，bpp=8×bytes/(352×288)；逐帧扩展组取 log1p(bpp) | 当前帧输出后，应用可直接统计输出包大小；不同容器/NAL 统计口径需保持一致 |
| encoder_bits | x265 CSV Bits | 当前帧输出后；仅保留取证，不用它替代包码率或混合不同开销定义 |
| source_luma_mean / source_luma_std | 未作 limited-range 截断的原始 8-bit Y 像素均值/总体标准差 | 编码前可计算，仍有像素遍历代价；仅用于扩展特征组 |

CU 统计的分母是该帧 **totalCu**，不是像素面积。尺寸不同的 CU 在该统计中各算一个单元。先读取全帧统计，再将四舍五入导致的 100% 附近微小误差归一化；后续另一组同名列是每深度 PU 统计，不混用。所有值单位为百分数 0–100，而非 0–1。

视频级使用：平均 SI/TI、帧 QP 的均值及总体标准差、每帧 intra/skip_merge 比例的算术平均。后者不是把全段所有 CU 计数汇总后的加权比例。扩展组另外使用帧 bpp 的总体标准差、原始帧亮度均值的平均、原始帧亮度标准差的平均。另保存 I/P/B 的平均 QP 与帧类型比例供审计；该配置的 I/P/B 比例固定为 2%/20%/78%，因此未作为视频级变量输入。

闭合 GOP 在 IDR 后令 POC 重新从零计数。提取器按每个 I-SLICE 的 GOP 偏移恢复全局显示帧号，并验证 0…149 无缺漏、与原表帧类型一一对应；CSV 行顺序本身是编码顺序，不能直接与显示顺序 PSNR 逐行合并。

明确排除：PSNR、SSIM、MSE、Avg Luma/Chroma Distortion、重建相关能量、残差能量、视频标识、编码 SHA-256、另一档码率的质量。编码器 CSV 中的 Avg Luma Distortion 已使用源图像与量化重建图像误差，把它用于本任务会把目标信息直接带入预测。

参考：[x265 CSV 文档](https://x265.readthedocs.io/en/master/cli.html#logging-statistic-options)、[固定版本 CU 计数实现](https://github.com/Multicorewareinc/x265/blob/8be7dbf81/source/encoder/frameencoder.cpp)、[固定版本 frameStats->qp 赋值](https://github.com/Multicorewareinc/x265/blob/8be7dbf81/source/encoder/encoder.cpp)。
