# fix-python-path.ps1
# Automatically adds Python Scripts folder to user PATH

# Get Python user base
$userBase = python -m site --user-base
$scriptsPath = Join-Path $userBase "Scripts"

# Check if path exists
if (-Not (Test-Path $scriptsPath)) {
    Write-Host "Scripts folder not found at $scriptsPath"
    exit 1
}

# Get current user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Add scripts path if not already present
if ($userPath -notlike "*$scriptsPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$scriptsPath", "User")
    Write-Host "Added $scriptsPath to user PATH successfully!"
} else {
    Write-Host "$scriptsPath is already in PATH."
}

Write-Host "You may need to restart PowerShell for changes to take effect."
Write-Host "After restart, test by running:"
Write-Host "pip --version"
Write-Host "streamlit hello"
