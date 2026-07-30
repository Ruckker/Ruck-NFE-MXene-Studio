# 中文：`build_windows.ps1` 的配置、依赖或构建说明。
# English: Configuration, dependency, or build instructions for `build_windows.ps1`.
# Author: Ruck
# Generated: 2026-07-30 08:42:34 Asia/Shanghai
# 提示 / Tip: 修改依赖、路径或阈值后，请重新运行测试与冒烟验证。
# Re-run tests and smoke validation after changing dependencies, paths, or thresholds.
param(
    [string]$Python = "F:\model\.venv-exe\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Spec = Join-Path $PSScriptRoot "NFE_MXene_Studio_1_0.spec"
$ReleaseName = "NFE_MXene_Studio_1_0"
$ReleaseDirectory = Join-Path $ProjectRoot "dist\$ReleaseName"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$required = @(
    (Join-Path $PSScriptRoot "models\nfe_predictor.pt"),
    (Join-Path $PSScriptRoot "models\mxene_generator.pt"),
    (Join-Path $PSScriptRoot "resources\surface_geometry_summary.json")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required model resource is missing: $path"
    }
}
if (Test-Path -LiteralPath $ReleaseDirectory) {
    throw "Release directory already exists; no files were overwritten: $ReleaseDirectory"
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --noconfirm $Spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$Executable = Join-Path $ReleaseDirectory "$ReleaseName.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Build completed without the expected executable: $Executable"
}
Write-Host "Built successfully: $Executable"
