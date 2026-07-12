Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -ne "" }
if ($proc) {
    $hwnd = $proc.MainWindowHandle
    $bmp = New-Object System.Drawing.Bitmap(800, 600)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen(0, 0, 0, 0, $bmp.Size, [System.Drawing.CopyPixelOperation]::SourceCopy)
    $bmp.Save("e:\xiangmu\aiwork\sjxm\canvas_screenshot.png")
    Write-Host "Screenshot saved"
} else {
    Write-Host "No window found"
}
