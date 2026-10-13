param(
    [string]$BuildDirectory = 'build/windows',
    [string]$CudaArchitectures = '89',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'Enter-BuildEnvironment.ps1')
if (-not $env:OpenCV_DIR) { throw 'OpenCV is missing from build/toolchain/opencv.' }
$buildPath = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $projectRoot $BuildDirectory }
& cmake -S $projectRoot -B $buildPath -G 'NMake Makefiles' `
    -DCMAKE_BUILD_TYPE=Release -DVINYLIZER_ENABLE_GUI=OFF `
    -DVINYLIZER_BUILD_TESTS=ON -DVINYLIZER_TEST_CUDA=ON `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
    -DCMAKE_CUDA_FLAGS=--use-local-env "-DOpenCV_DIR=$env:OpenCV_DIR"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed: $LASTEXITCODE" }
& cmake --build $buildPath --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
# Keep the built CLI usable outside the activated development shell.
$runtimeDlls = @(Get-ChildItem "$env:OpenCV_DIR/../bin/opencv_world*.dll" |
    Where-Object { $_.BaseName -match '^opencv_world\d+$' })
$runtimeDlls += Get-Item "$env:CUDA_PATH/bin/cudart64_12.dll"
foreach ($dll in $runtimeDlls) { Copy-Item -LiteralPath $dll.FullName -Destination $buildPath -Force }
if (-not $SkipTests) {
    & ctest --test-dir $buildPath -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
}
