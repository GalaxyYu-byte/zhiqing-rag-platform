param(
    [ValidateSet('backend', 'frontend', 'worker', 'graph')]
    [string]$Service
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Service) {
    Set-Location -LiteralPath $ProjectRoot

    switch ($Service) {
        'backend' { uv run python -m zhiqing_rag.main }
        'frontend' {
            Set-Location -LiteralPath (Join-Path $ProjectRoot 'frontend')
            npm run dev
        }
        'worker' { uv run python -m zhiqing_rag.workers.ingestion }
        'graph' { uv run python -m zhiqing_rag.workers.graph }
    }

    exit $LASTEXITCODE
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '找不到 uv。请先安装 uv，并确保它已加入 PATH。'
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw '找不到 npm。请先安装 Node.js，并确保它已加入 PATH。'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'frontend\node_modules'))) {
    throw '前端依赖尚未安装。请先运行：cd frontend; npm ci'
}

$PowerShellExe = (Get-Process -Id $PID).Path
$ScriptPath = $MyInvocation.MyCommand.Path
$Services = @('backend', 'frontend', 'worker')
$EnvPath = Join-Path $ProjectRoot '.env'
$GraphEnabled = $env:NEO4J_ENABLED -eq 'true'
if ((-not $GraphEnabled) -and (Test-Path -LiteralPath $EnvPath)) {
    $GraphEnabled = [bool](Select-String -LiteralPath $EnvPath -Pattern '^\s*NEO4J_ENABLED\s*=\s*"?true"?\s*$' -Quiet)
}
if ($GraphEnabled) { $Services += 'graph' }

foreach ($Name in $Services) {
    $Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$ScriptPath`" -Service $Name"
    Start-Process -FilePath $PowerShellExe -ArgumentList $Arguments -WorkingDirectory $ProjectRoot
}

Write-Host '已启动后端、前端和文档 Worker。'
if ($GraphEnabled) { Write-Host '已单独启动 Graph Worker。' }
Write-Host '前端：http://127.0.0.1:5173'
Write-Host '后端 API 文档：http://127.0.0.1:8000/docs'
