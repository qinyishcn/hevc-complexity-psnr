# H.265 内容复杂度与同码率 PSNR 实验

日期：2026-10-08。以下保留原实验的设计与更新记录；公开仓库是原研究目录的独立导出。

## 目标与预先确定的设计

调研 SI/TI 后，使用 Xiph 公开原始 YUV4MPEG 测试素材，测量相同平均视频码率下的解码重建 PSNR，绘制逐帧和序列级散点、线性拟合、二次拟合，检验相关方向。不预设正相关成立。

- 20 个序列，按名称预先选取以覆盖人物、交通、运动、水流和纹理；每个只取前 150 帧，不按 PSNR 挑选。
- 原始 CIF 352×288，8 bit YUV420；统一将帧序列播放为 30 fps，不进行插帧、丢帧或空间缩放。原始帧率保留在清单中。
- 经典 P.910 (04/2008) SI/TI：原始亮度码值上的 Sobel 幅度标准差和相邻帧差标准差。SI 去掉一圈边界，TI 全画面；总体标准差 ddof=0。第一帧 TI 缺失，不填零。明确区分经典定义与新版 SDR/HDR 扩展。
- x265 medium + tune psnr；闭合 GOP=60、固定 B 帧策略、禁用自适应场景切换；150/300/600 kb/s 两遍 ABR。以 ffprobe 包大小总和计算实际视频码率，必要时迭代目标值，误差验收 <=1%。不填充无用数据伪造等码率。
- 逐帧比较显示顺序的重建 YUV，主指标 PSNR-Y，附 PSNR-YUV。序列 PSNR 对 MSE 先取平均再转 dB，另保留逐帧 dB 均值。
- 每个码率分别估计 Pearson、Spearman、bootstrap 95% CI 和序列级置换检验。按预先指定的 SI-max/TI-max 六个主检验作 Holm 校正。均值及联合回归为补充，不当作新的独立重复实验。
- 逐帧结果用于展示，不用几千个帧的普通 p 值冒充独立样本证据。补充按序列聚类的回归、去掉 I 帧和序列内中心化分析。

## 顺序与验收

1. 保存标准/工具/素材来源与获取日期；下载原文件的完整前 150 帧，记录 SHA256、HTTP Range 和源头信息。
2. 先验证度量：静止序列 TI=0、亮度整体平移 TI=0、有符号帧差无 uint8 回绕、已知 MSE 的 PSNR；与 FFmpeg siti/psnr 交叉核验。
3. 执行 60 次最终编码，保存全部命令和日志、两遍率控文件、实际包码率、帧计数与帧类型；不符合码率要求则修正或明确失败，不静默丢样本。
4. 输出 frame_metrics.csv、sequence_metrics.csv、统计 CSV/JSON、PNG/SVG 科学图、中文报告与可浏览 HTML。
5. 验收全部预选序列、所有目标码率、相同帧数/格式/尺寸、码率误差、有限度量值、图形视觉检查与报告数字一致。

## 来源

### 完整标准阅读后的更新（在任何编码结果生成前）

2023 版 7.8.4 改为推荐均值，并在 7.8.1 增加 SDR/HDR 亮度预处理。因此六个主检验改为 **2023 流程 SI-mean/TI-mean × 三码率**；经典原始码值算法、旧版最大值均作为敏感性分析。2023 流程明确使用 limited 8 bit → [0,1]（截断越界码值）→ BT.1886 完整黑电平偏移公式（Lw=300、Lb=0.01 cd/m²、gamma=2.4）→ BT.2100 PQ → SI/TI → ×255。报告不得将此与 FFmpeg 经典 siti 或 VQEG 默认简化显示模型混为一谈。

单连接实测 Xiph 约 20 kB/s；在不改变来源与像素的前提下改为 32 个有界 HTTP Range 下载，逐块检查 Content-Range 和长度并落盘，以避免长连接超时丢失进度。

### 完整素材审计后的敏感性补充

源 Y4M 不标记 range，部分素材包含大量超出16–235的亮度码值。除上述 limited-range 主分析外，新增 full-range 解释（Y/255、相同 BT.1886/PQ 参数）的 SI/TI 和统计作为敏感性分析，不变更原定六个主检验。编码与 PSNR 保持原始像素不变。以全部256码值核对两种解释的传递函数，并在报告中明确范围元数据的缺失。

- ITU-T P.910 (04/2008), 5.3.1/5.3.2 与 Annex A：https://www.itu.int/rec/dologin_pub.asp?id=T-REC-P.910-200804-S!!PDF-E&lang=e&type=items
- ITU-T P.910 (10/2023)：https://www.itu.int/rec/dologin_pub.asp?id=T-REC-P.910-202310-S!!PDF-E&lang=e&type=items
- Xiph 测试集及逐素材版权说明：https://media.xiph.org/video/derf/
- x265 rate control/tune/pass：https://x265.readthedocs.io/en/master/cli.html
- FFmpeg PSNR：https://ffmpeg.org/ffmpeg-filters.html#psnr
- FFmpeg 8.1 SI/TI 实现：https://ffmpeg.org/doxygen/8.1/vf__siti_8c_source.html

Xiph 是公开测试素材集合，并非统一 CC/OSI 开放许可证的数据集。不得声称所有素材已授权任意商业用途。原文件只本地用于这次编解码评估，不发布素材。
