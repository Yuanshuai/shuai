#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包脚本 - 使用 PyInstaller 打包应用程序
排除不需要的模型文件以减小体积
"""

import os
import sys
import shutil
import subprocess


def clean_build_dirs():
    """清理构建目录"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"清理 {dir_name} 目录...")
            shutil.rmtree(dir_name)
    print("清理完成")


def build():
    """执行打包"""
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 清理旧构建
    clean_build_dirs()
    
    # 执行 PyInstaller（使用极简版配置）
    spec_file = os.path.join(project_root, "packaging", "qt_minimal.spec")
    
    print("=" * 60)
    print("开始打包...")
    print(f"使用 spec 文件: {spec_file}")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        spec_file,
        "--clean",
        "--noconfirm"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ 打包成功!")
        print("=" * 60)
        
        # 显示输出目录大小
        dist_dir = os.path.join(project_root, "dist", "Convert")
        if os.path.exists(dist_dir):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dist_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            size_mb = total_size / (1024 * 1024)
            print(f"输出目录: {dist_dir}")
            print(f"总大小: {size_mb:.2f} MB")
            
            # 列出包含的模型文件
            print("\n包含的模型文件:")
            for root, dirs, files in os.walk(dist_dir):
                for file in files:
                    if file.endswith('.onnx'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        rel_path = os.path.relpath(file_path, dist_dir)
                        print(f"  - {rel_path}: {file_size:.2f} MB")
    else:
        print("\n" + "=" * 60)
        print("❌ 打包失败!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    build()
