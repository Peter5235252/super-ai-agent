Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
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
    [DllImport("user32.dll")] public static extern bool DestroyIcon(IntPtr h);
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

$repoRoot = $PSScriptRoot
if (-not $repoRoot -or -not (Test-Path (Join-Path $repoRoot "Cargo.toml"))) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
}
if (-not $repoRoot -or -not (Test-Path (Join-Path $repoRoot "Cargo.toml"))) {
    $repoRoot = (Get-Location).Path
}
$outDir = Join-Path $repoRoot "src-tauri/icons"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# 512 source for PNGs
$bmp512 = New-IconBitmap 512
$bmp512.Save((Join-Path $outDir "icon.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp128 = New-Object System.Drawing.Bitmap($bmp512, (New-Object System.Drawing.Size(128, 128)))
$bmp128.Save((Join-Path $outDir "128x128.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp32 = New-Object System.Drawing.Bitmap($bmp512, (New-Object System.Drawing.Size(32, 32)))

# ICO must be BMP-backed for RC 3.00 — build from a 32px HICON, not a PNG.
$bmp32.Save((Join-Path $outDir "32x32.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$h = $bmp32.GetHicon()
try {
    $icon = [System.Drawing.Icon]::FromHandle($h)
    try {
        $fs = [System.IO.File]::Create((Join-Path $outDir "icon.ico"))
        try {
            $icon.Save($fs)
        } finally {
            $fs.Dispose()
        }
    } finally {
        $icon.Dispose()
    }
} finally {
    [GfxExt]::DestroyIcon($h) | Out-Null
    $bmp32.Dispose()
}
$bmp128.Dispose()
$bmp512.Dispose()
Get-ChildItem $outDir | Select-Object Name, Length | Format-Table -AutoSize
