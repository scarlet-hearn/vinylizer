param([string]$ToolchainRoot = (Join-Path $PSScriptRoot '../build/toolchain'))

$ErrorActionPreference = 'Stop'
$ToolchainRoot = (Resolve-Path -LiteralPath $ToolchainRoot).Path
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vsInstall = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) { throw 'Visual Studio C++ tools were not found.' }
$msvcVersion = (Get-Content "$vsInstall/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt").Trim()
$msvcRoot = "$vsInstall/VC/Tools/MSVC/$msvcVersion"
$sdkRoot = "$ToolchainRoot/winsdk/c"
$sdkVersion = '10.0.26100.0'
$sdkLibraries = "$ToolchainRoot/winsdk-x64/c"
$cudaRoot = "$ToolchainRoot/cuda"
$required = @(
    "$msvcRoot/bin/Hostx64/x64/cl.exe",
    "$sdkRoot/Include/$sdkVersion/ucrt/math.h",
    "$sdkLibraries/um/x64/kernel32.lib",
    "$ToolchainRoot/cmake/bin/cmake.exe",
    "$cudaRoot/bin/nvcc.exe"
)
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing build dependency: $file" }
}

# This SDK is installed from Microsoft's architecture-specific NuGet packages;
# its headers and libraries live in separate directories. Set this session only.
$env:INCLUDE = "$msvcRoot/include;$sdkRoot/Include/$sdkVersion/ucrt;$sdkRoot/Include/$sdkVersion/shared;$sdkRoot/Include/$sdkVersion/um;$sdkRoot/Include/$sdkVersion/winrt"
$env:LIB = "$msvcRoot/lib/x64;$sdkLibraries/ucrt/x64;$sdkLibraries/um/x64"
$env:PATH = "$ToolchainRoot/cmake/bin;$cudaRoot/bin;$msvcRoot/bin/Hostx64/x64;$sdkRoot/bin/$sdkVersion/x64;$env:PATH"
$env:VSINSTALLDIR = "$vsInstall/"
$env:VCToolsInstallDir = "$msvcRoot/"
$env:WindowsSdkDir = "$sdkRoot/"
$env:WindowsSDKVersion = "$sdkVersion/"
$env:CUDA_PATH = $cudaRoot
$env:CUDACXX = "$cudaRoot/bin/nvcc.exe"

$opencvConfig = "$ToolchainRoot/opencv/build/x64/vc16/lib"
if (Test-Path -LiteralPath "$opencvConfig/OpenCVConfig.cmake") {
    $env:OpenCV_DIR = $opencvConfig
    $env:PATH = "$ToolchainRoot/opencv/build/x64/vc16/bin;$env:PATH"
}
Write-Host "Build environment ready: MSVC $msvcVersion, Windows SDK $sdkVersion, CUDA $cudaRoot"
