# Pico GPT CLI Client Launcher for Windows PowerShell
param(
    [string]$Model = "",
    [string]$Device = "auto", 
    [int]$MaxTokens = 100,
    [double]$Temperature = 0.8,
    [int]$TopK = 20,
    [string]$Prompt = "",
    [switch]$Help
)

function Write-ColorOutput([String] $ForegroundColor, [String] $Text) {
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Text
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

function Write-Success([String] $Text) { Write-ColorOutput Green $Text }
function Write-Error([String] $Text) { Write-ColorOutput Red $Text }
function Write-Warning([String] $Text) { Write-ColorOutput Yellow $Text }
function Write-Info([String] $Text) { Write-ColorOutput Cyan $Text }

if ($Help) {
    Write-Output "Usage: .\run_cli.ps1 [OPTIONS]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Model <path>        Path to model file"
    Write-Output "  -Device <device>     Device: cpu, cuda, auto (default: auto)"
    Write-Output "  -MaxTokens <int>     Maximum tokens (default: 100)"
    Write-Output "  -Temperature <float> Temperature (default: 0.8)"
    Write-Output "  -TopK <int>          Top-k sampling (default: 20)"
    Write-Output "  -Prompt <string>     Single prompt mode"
    Write-Output "  -Help                Show this help"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output "  .\run_cli.ps1"
    Write-Output "  .\run_cli.ps1 -Prompt `"Hello world`""
    Write-Output "  .\run_cli.ps1 -Model model.pt -MaxTokens 50"
    exit 0
}

Write-Info "==============================================="
Write-Info "   Pico GPT CLI Client Launcher (PowerShell)"
Write-Info "==============================================="
Write-Output ""

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not installed or not in PATH"
    exit 1
}
Write-Success "Python found: $pythonVersion"

# Check cli_client.py
if (-not (Test-Path "cli_client.py")) {
    Write-Error "cli_client.py not found. Run from pico-gpt directory."
    exit 1
}
Write-Success "CLI client found"

# Check model files
$modelFiles = Get-ChildItem -Path "*.pt" -ErrorAction SilentlyContinue
if ($modelFiles.Count -eq 0) {
    Write-Warning "No model files (.pt) found"
    Write-Warning "Train a model first using train_large.py or train_small.py"
} else {
    Write-Success "Found $($modelFiles.Count) model file(s)"
    foreach ($file in $modelFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 1)
        $fileName = $file.Name
        Write-Output "    $fileName - $sizeMB MB"
    }
}

Write-Output ""
Write-Info "Starting Pico GPT CLI Client..."
Write-Output ""

# Build arguments
$pythonArgs = @()
if ($Model) { $pythonArgs += "--model", $Model }
if ($Device -ne "auto") { $pythonArgs += "--device", $Device }
if ($MaxTokens -ne 100) { $pythonArgs += "--max-tokens", $MaxTokens }
if ($Temperature -ne 0.8) { $pythonArgs += "--temperature", $Temperature }
if ($TopK -ne 20) { $pythonArgs += "--top-k", $TopK }
if ($Prompt) { $pythonArgs += "--prompt", $Prompt }

# Run the CLI client
& python cli_client.py @pythonArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Program exited with error code $LASTEXITCODE"
}
