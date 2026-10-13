# Windows 编译验证记录

验证日期：2026-10-13。Release，x64，RTX 4060 Laptop GPU（8 GB），CUDA 架构 `sm_89`。

## 工具与安装位置

| 工具 | 版本 | 安装方式 / 位置 |
| --- | --- | --- |
| CMake | 3.31.12 | 官方 ZIP，`build/toolchain/cmake` |
| nvcc | 12.9.86（CUDA 12.9 Update 1） | NVIDIA 官方 nvcc、cudart、CCCL 分发包，`build/toolchain/cuda` |
| cl | 19.44.35215，工具集目录 14.44.35207 | 复用已有 Visual Studio Community 2022 |
| Windows SDK | 10.0.26100.0，NuGet 包 10.0.26100.1882 | Microsoft 官方 CPP 和 CPP.x64 包，`build/toolchain/winsdk`、`winsdk-x64` |
| OpenCV | 4.12.0 | 官方 Windows 发布包，`build/toolchain/opencv` |
| Qt | 6.8.3 MSVC 2022 x64 | 官方 qtbase 分发包，`build/toolchain/qt` |

新增组件均放在项目本地，未修改系统 PATH，也未更新显卡驱动。`Enter-BuildEnvironment.ps1` 只设置当前 PowerShell 会话的 PATH、INCLUDE、LIB 等变量。SDK 的头文件和库来自独立的官方 NuGet 包，因此 nvcc 使用 `--use-local-env` 复用已设置的环境。

来源：[CMake](https://cmake.org/download/)、[NVIDIA CUDA 12.9.1 清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)、[Windows SDK CPP](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP/10.0.26100.1882)、[Windows SDK CPP.x64](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP.x64/10.0.26100.1882)、[OpenCV 4.12.0](https://github.com/opencv/opencv/releases/tag/4.12.0)。

CMake、CUDA、OpenCV 下载核对官方 SHA-256；SDK CPP 核对官方响应中的 SHA-512。首次 OpenCV 下载不完整，未使用；重新下载的文件长度 186838616 字节，SHA-256 为 `b753b14d880b9bc8d89d6acd3b665c040baec0211078435432fcae117db707af`，与发布摘要一致。

## 构建

```powershell
./tools/Build-Windows.ps1
```

该脚本使用 NMake、Release、CUDA 架构 89，构建命令行程序和所有测试；将 OpenCV、CUDA 运行时 DLL 复制到主程序旁。产物为 `build/windows/vinylizer.exe`，Haar 模型位于相邻的 `assets/haarcascades`。工具与产物均被 Git 忽略。

构建发现并修复：

- CUDA 测试显式添加 `<string>`，解决 nvcc 无法解析 `std::to_string`。
- 主流程的 `small` 变量更名，解决 Windows SDK 同名宏冲突。
- MSVC 和 nvcc 的主机编译统一使用 UTF-8，避免本机代码页 936 的源文件编码警告。

MSVC 在受限沙箱中的编译探测曾报 C1902；相同工具链在正常本机环境下编译成功。构建仍存在原有的 CRT 弃用提示与少量类型转换警告，未将它们当作测试失败。

## 测试结果

`ctest --test-dir build/windows --output-on-failure`：**6/6 通过，无跳过**。

| 测试 | 覆盖范围 | 结果 |
| --- | --- | --- |
| pixel_loss | 加权 RGB、Alpha、24 项数值差分 | 通过 |
| fitting_math | 648 项多图层梯度差分、遮挡、残差分散与方向 | 通过 |
| renderer_gradient | 真实 GPU 的整图/分块梯度与遮挡缓存复用 | 通过 |
| safety | JSON、图层表、模拟进程内存、输出错误 | 通过 |
| image_input | 缩放、长宽比、位深、预乘 Alpha | 通过 |
| facial_saliency | 五官权重、漏检补偿、归一化、模型资源 | 通过 |

完整流程另使用两张 96×80 的合成输入，优化长边 64，32 图层，2 个循环（400 次更新），学习率 `0.1,0.01`：

| 输入 | 导出图层 | 最佳加权 MSE | 结果 |
| --- | ---: | ---: | --- |
| RGB | 32 | 1063.068 | JSON 和 96×80 PNG 预览成功 |
| 半透明 RGBA | 32 | 130.870 | JSON 和 96×80 PNG 预览成功，20 层 Alpha 小于 255 |

验证了导出参数均有限、RGBA 字节范围正确、图层数保持 32、预览可解码且尺寸正确。输入和输出保存在 `build/smoke`。这些合成用例验证流程可运行；上述 MSE 不是新旧算法的 A/B 对比，也不是自然图片画质提升的量化结果。

## Qt 界面验证

Qt 来自 [Qt 6.8.3 官方仓库](https://download.qt.io/online/qtsdkrepository/windows_x86/desktop/qt6_683/qt6_683/qt.qt6.683.win64_msvc2022_64/)，qtbase 包已核对官方 SHA-1：`ab0bd2551618f7128ef30c6d74fe45b6a76e2296`。无需 Qt Creator；qtbase 包含本项目所需的 Widgets、OpenGL、OpenGLWidgets、Concurrent 及 windeployqt。

```powershell
./tools/Build-Windows.ps1 -Gui
# 可选：-QtDirectory C:/Qt/6.8.3/msvc2022_64
./build/windows-qt/vinylizer.exe
```

独立目录 `build/windows-qt` 包含 Qt、OpenCV、CUDA DLL 和 Windows 平台插件，可在普通终端直接启动。`Build-Windows.ps1 -Gui` 实际执行成功，CTest **8/8 通过，无跳过**（4.69 秒）。在原 6 项测试基础上增加：

- `gui_config`：完整文本解析，拒绝小数整数、空值、尾随字符、非有限数与溢出，验证学习率关系及循环预算边界。
- `gui_preview`：真实 GPU 渲染，验证 JSON 画布尺寸与原点、越界图层裁剪、RGB 通道、透明合成、图层顺序、中文路径、无尺寸旧文件、2048 像素上限及错误输入。

实际桌面交互使用 1200×800 的 Qt 窗口，输入文件和日志位于 `build/gui-smoke`：

| 场景 | 验证结果 |
| --- | --- |
| 启动、中文标签、文件对话框、制作/导入切页 | 正常，无 Qt 平台插件或 OpenGL 初始化错误 |
| RGB 96×80，150 层，2 循环 | 进度条、MSE 曲线和实时预览正常，完整导出 150 层 |
| 中文文件名 RGBA，150 层，2 循环 | 正常读取和导出，103 层 Alpha 小于 255 |
| 150 层，200 循环，中途停止 | 显示“已停止并保存结果”，导出完整 150 层及最佳参数 |
| JSON 导入预览、占位层数不匹配 | 异步预览正常加载，错误消息可见，未进行游戏写入 |
| 修复后 RGB 150 层、25 循环 | 正常完成；完成页按原画布显示，同一份旧 JSON 的导入预览也恢复正确构图 |
| 修复后数值输入 | 图层数量拒绝小数点；导入层数 200000 时按钮禁用，点击不启动任务 |

验证过程中修复两类问题：完成页和导入页以前按图层包围盒扩大画布，导致预览缩小并显示原画布外的图层；现在优先使用 JSON 的 width/height 和原点，旧版无尺寸文件保留包围盒兼容。自绘输入框以前用 `stoi` 接受部分文本，空层数会沿用旧值；现在优化配置完整校验，注入按钮和启动动作均验证当前层数文本。

当前自绘输入框仍仅支持逐键编辑、左右键和 Backspace，不支持 Ctrl+A/粘贴；Tab 用于切换页面。透明输入在优化中的 RGB 实时帧与完成后的 Alpha 合成预览可能不同。以上合成输入用于流程验证，未替代真实图片集画质评估。

## 尚未覆盖

- 真实 FH6 游戏进程的导入。
- 新旧算法在同一真实图片集上的质量和耗时比较。
