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

## 尚未覆盖

- Qt GUI 的编译及交互测试（本次未安装 Qt）。
- 真实 FH6 游戏进程的导入。
- 新旧算法在同一真实图片集上的质量和耗时比较。
