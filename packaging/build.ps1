$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root "venv\Scripts\python.exe"

& $python -m PyInstaller --noconfirm --clean (Join-Path $root "packaging\qt.spec")
& $python -m PyInstaller --noconfirm --clean (Join-Path $root "packaging\lin.spec")

# Clean up unnecessary files
$qtInternal = Join-Path $root "dist\qt_app\_internal"
$linInternal = Join-Path $root "dist\lin_server\_internal"

foreach ($dir in @($qtInternal, $linInternal)) {
    if (Test-Path $dir) {
        Write-Host "Cleaning $dir..."
        
        # Delete .lib files (exclude .libs directories)
        Get-ChildItem -Path $dir -Filter "*.lib" -Recurse -ErrorAction SilentlyContinue | 
            Where-Object { $_.FullName -notlike "*.libs*" } | 
            Remove-Item -Force
        
        # Delete .h header files
        Get-ChildItem -Path $dir -Filter "*.h" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete __pycache__ directories
        Get-ChildItem -Path $dir -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        
        # Delete .dist-info directories
        Get-ChildItem -Path $dir -Directory -Filter "*.dist-info" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        
        # Delete ffmpeg video codecs (PDF/OCR doesn't need video processing)
        Get-ChildItem -Path $dir -Filter "*ffmpeg*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete PIL AVIF support (not commonly used)
        Get-ChildItem -Path $dir -Filter "*avif*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete PIL Tkinter support (not needed)
        Get-ChildItem -Path $dir -Filter "*imagingtk*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete PIL WebP support (PDF/OCR doesn't need WebP)
        Get-ChildItem -Path $dir -Filter "*webp*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete tcl/tk directories if exist
        Get-ChildItem -Path $dir -Directory -Filter "tcl*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Get-ChildItem -Path $dir -Directory -Filter "tk*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        
        # Delete OpenCV haarcascade files (face detection models, not needed for PDF/OCR)
        Get-ChildItem -Path $dir -Filter "haarcascade*" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
        
        # Delete rapidocr font file (FZYTK.TTF, not needed for OCR)
        Get-ChildItem -Path $dir -Filter "FZYTK.TTF" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
    }
}

Write-Host "Done."
Write-Host "Qt:  $(Join-Path $root 'dist\qt_app\qt_app.exe')"
Write-Host "Lin: $(Join-Path $root 'dist\lin_server\lin_server.exe')"
