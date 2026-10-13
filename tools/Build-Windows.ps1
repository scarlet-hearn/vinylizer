param(
    [string]$BuildDirectory,
    [string]$CudaArchitectures = '89',
    [switch]$SkipTests,
    [switch]$Gui,
    [string]$QtDirectory = 'build/toolchain/qt'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'Enter-BuildEnvironment.ps1')
if (-not $env:OpenCV_DIR) { throw 'OpenCV is missing from build/toolchain/opencv.' }
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = if ($Gui) { 'build/windows-qt' } else { 'build/windows' }
}
$buildPath = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $projectRoot $BuildDirectory }
$guiMode = if ($Gui) { 'ON' } else { 'OFF' }
$configureArguments = @(
    '-S', $projectRoot, '-B', $buildPath, '-G', 'NMake Makefiles',
    '-DCMAKE_BUILD_TYPE=Release', "-DVINYLIZER_ENABLE_GUI=$guiMode",
    '-DVINYLIZER_BUILD_TESTS=ON', '-DVINYLIZER_TEST_CUDA=ON',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
    '-DCMAKE_CUDA_FLAGS=--use-local-env', "-DOpenCV_DIR=$env:OpenCV_DIR"
)
if ($Gui) {
    $qtPath = if ([IO.Path]::IsPathRooted($QtDirectory)) { $QtDirectory } else { Join-Path $projectRoot $QtDirectory }
    $qtPath = (Resolve-Path -LiteralPath $qtPath).Path
    $qtCmakePath = Join-Path $qtPath 'lib/cmake/Qt6'
    if (-not (Test-Path -LiteralPath (Join-Path $qtCmakePath 'Qt6Config.cmake'))) {
        throw "Qt6 CMake configuration is missing from $qtCmakePath. Specify -QtDirectory with a Qt installation prefix."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $qtPath 'bin/windeployqt.exe'))) {
        throw "Qt deployment tool is missing from $qtPath/bin."
    }
    $env:PATH = "$(Join-Path $qtPath 'bin');$env:PATH"
    $configureArguments += "-DCMAKE_PREFIX_PATH=$qtPath", "-DQt6_DIR=$qtCmakePath"
}
& cmake @configureArguments
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed: $LASTEXITCODE" }
& cmake --build $buildPath --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
# Keep both entry points usable outside the activated development shell.
$runtimeDlls = @(Get-ChildItem "$env:OpenCV_DIR/../bin/opencv_world*.dll" |
    Where-Object { $_.BaseName -match '^opencv_world\d+$' })
$runtimeDlls += Get-Item "$env:CUDA_PATH/bin/cudart64_12.dll"
foreach ($dll in $runtimeDlls) { Copy-Item -LiteralPath $dll.FullName -Destination $buildPath -Force }
if (-not $SkipTests) {
    & ctest --test-dir $buildPath -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
}
