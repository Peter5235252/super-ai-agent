Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System.Drawing;
using System.Drawing.Drawing2D;
public static class GfxExt {
    public static void FillRoundedRectangle(Graphics g, Brush b, int x, int y, int w, int h, int r) {
        GraphicsPath p = new GraphicsPath();
        p.AddArc(x, y, r, r, 180, 90);
        p.AddArc(x + w - r, y, r, r, 270, 90);
        p.AddArc(x + w - r, y + h - r, r, r, 0, 90);
        p.AddArc(x, y + h - r, r, r, 90, 90);
        p.CloseFigure();
        g.FillPath(b, p);
        p.Dispose();
    }
}
"@ -ReferencedAssemblies System.Drawing

function New-IconBitmap([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $g.Clear([System.Drawing.Color]::FromArgb(255, 11, 14, 20))
        $rect = New-Object System.Drawing.Rectangle(0, 0, $size, $size)
        $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
            $rect,
            [System.Drawing.Color]::FromArgb(255, 37, 99, 235),
            [System.Drawing.Color]::FromArgb(255, 124, 58, 237),
            [System.Drawing.Drawing2D.LinearGradientMode]::ForwardDiagonal
        )
        try {
            $margin = [int]($size * 0.06)
            [GfxExt]::FillRoundedRectangle($g, $brush, $margin, $margin, ($size - 2 * $margin), ($size - 2 * $margin), [int]($size * 0.22))
        } finally {
            $brush.Dispose()
        }
        $cx = $size / 2.0
        $r = $size * 0.26
        $pts = @(
            (New-Object System.Drawing.PointF($cx, ($cx - $r))),
            (New-Object System.Drawing.PointF(($cx + $r * 0.62), $cx)),
            (New-Object System.Drawing.PointF($cx, ($cx + $r))),
            (New-Object System.Drawing.PointF(($cx - $r * 0.62), $cx))
        )
        $fill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 230, 234, 242))
        try {
            $g.FillPolygon($fill, $pts)
        } finally {
            $fill.Dispose()
        }
    } finally {
        $g.Dispose()
    }
    return $bmp
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
$outDir = Join-Path $repoRoot "src-tauri/icons"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$bmp512 = New-IconBitmap 512
$bmp512.Save((Join-Path $outDir "icon.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp128 = New-Object System.Drawing.Bitmap($bmp512, (New-Object System.Drawing.Size(128, 128)))
$bmp128.Save((Join-Path $outDir "128x128.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp32 = New-Object System.Drawing.Bitmap($bmp512, (New-Object System.Drawing.Size(32, 32)))
$bmp32.Save((Join-Path $outDir "32x32.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp256 = New-Object System.Drawing.Bitmap($bmp512, (New-Object System.Drawing.Size(256, 256)))
$bmp256.Save((Join-Path $outDir "icon.ico"), [System.Drawing.Imaging.ImageFormat]::Icon)
$bmp256.Dispose()
$bmp32.Dispose()
$bmp128.Dispose()
$bmp512.Dispose()
Get-ChildItem $outDir | Select-Object Name, Length | Format-Table -AutoSize
