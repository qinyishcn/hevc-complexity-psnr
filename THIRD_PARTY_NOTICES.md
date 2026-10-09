# 第三方来源与许可

`sources/vqeg_siti.py` 原样保存自 VQEG/siti-tools，MIT License：

- 作者版权：Copyright (c) 2021 Werner Robitza, Lukas Krasula。
- 固定提交：`c537e33793266187aad87c859fd62a9de9033854`。
- 来源：[siti.py](https://github.com/VQEG/siti-tools/blob/c537e33793266187aad87c859fd62a9de9033854/src/siti_tools/siti.py)。
- 完整许可证：[sources/vqeg_license.txt](sources/vqeg_license.txt)。
- 来源与源文件 SHA-256：[sources/vqeg_provenance.json](sources/vqeg_provenance.json)。

验证脚本仅提取并执行其中独立的 BT.1886 与 PQ 数值函数，不导入视频 I/O 依赖。主实验采用带完整黑电平偏移的 BT.1886，不等同于参考包默认的简化显示封装。

Xiph 媒体、ITU-T 标准、FFmpeg 和 x265 各自受原权利人的许可约束。本仓库未收录 Xiph 视频或标准全文。VQEG 文件的 MIT 许可证仅适用于该第三方代码，不自动扩展至整个仓库。
