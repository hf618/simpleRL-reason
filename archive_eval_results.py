#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
from pathlib import Path

def archive_single_folder(source_dir_name, save_path, prefix, base_dir):
    """
    (内部函数) 查找、复制并压缩单个指定的评估结果文件夹。
    如果成功，返回生成的zip文件路径；如果失败，返回None。
    """
    print("\n" + "#"*60)
    print(f"▶️ 开始处理: {source_dir_name}")
    print("#"*60)

    # 1. 构建完整的源路径
    full_source_path = Path(base_dir) / source_dir_name
    print(f"[*] 源目录: {full_source_path}")

    # 2. 检查源路径是否存在
    if not full_source_path.is_dir():
        print(f"❌ 错误: 源目录 '{full_source_path}' 不存在或不是一个目录。")
        return None

    # 3. 确保目标路径是一个目录
    target_save_path = Path(save_path)
    if target_save_path.exists() and not target_save_path.is_dir():
        print(f"❌ 错误: 目标路径 '{target_save_path}' 已存在但不是一个目录。")
        return None
    target_save_path.mkdir(parents=True, exist_ok=True)
    
    # 4. 创建一个临时中转目录
    staging_dir = target_save_path / f"{source_dir_name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    # 5. 查找所有匹配前缀的目录并复制到中转目录
    found_folders = []
    for item in full_source_path.iterdir():
        if item.name.startswith(prefix) and item.is_dir():
            print(f"    - 找到并复制: {item.name}")
            shutil.copytree(item, staging_dir / item.name)
            found_folders.append(item.name)

    # 6. 如果一个都没找到，则报告失败并清理
    if not found_folders:
        print(f"⚠️ 警告: 在 '{full_source_path}' 中没有找到任何以 '{prefix}' 开头的目录。")
        shutil.rmtree(staging_dir) # 清理空的中转目录
        return None

    print(f"[*] 共复制了 {len(found_folders)} 个文件夹到中转目录。")

    # 7. 将中转目录压缩成zip文件
    zip_filename_base = target_save_path / source_dir_name
    print(f"[*] 正在创建ZIP压缩包: {zip_filename_base}.zip")
    
    try:
        archive_path = shutil.make_archive(
            base_name=zip_filename_base,
            format='zip',
            root_dir=staging_dir
        )
    except Exception as e:
        print(f"❌ 错误: 创建ZIP文件时发生错误: {e}")
        return None
    finally:
        # 8. 确保清理中转目录
        print(f"[*] 正在清理临时中转目录...")
        shutil.rmtree(staging_dir)

    print(f"✅ 成功! 压缩包已保存至: {archive_path}")
    return archive_path


def main():
    parser = argparse.ArgumentParser(
        description="将一个或多个实验目录下的评估结果(eval_results)文件夹分别打包压缩，并存放到同一个目标路径下。",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [OPTIONS] SOURCE_DIR_1 [SOURCE_DIR_2 ...] DESTINATION_PATH"
    )
    
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="一个或多个源实验目录名，最后跟上目标保存路径。\n"
             "例如: dir1 dir2 /path/to/save"
    )
    
    parser.add_argument(
        "--prefix",
        default="eval_results",
        help="要查找的子目录的前缀。(默认: 'eval_results')"
    )
    
    parser.add_argument(
        "--base-dir",
        default="/home/root1/Fanding/simpleRL-reason/custom/checkpoint/qwen",
        help="包含所有实验目录的基础路径。\n"
             "(默认: '/home/root1/Fanding/simpleRL-reason/custom/checkpoint/qwen')"
    )

    args = parser.parse_args()
    
    if len(args.paths) < 2:
        print("错误: 您必须至少提供一个源目录和一个目标路径。")
        parser.print_help()
        sys.exit(1)

    source_dir_names = args.paths[:-1]
    save_path = args.paths[-1]

    print(f"[*] 任务开始: 共 {len(source_dir_names)} 个源目录, 目标路径为 '{save_path}'")
    
    successful_tasks = []
    failed_tasks = []

    for source_name in source_dir_names:
        archive_path = archive_single_folder(source_name, save_path, args.prefix, args.base_dir)
        if archive_path:
            successful_tasks.append(source_name)
        else:
            failed_tasks.append(source_name)
    
    print("\n" + "="*60)
    print("                ✨ 所有任务处理完毕 ✨")
    print("="*60)
    print(f"  成功处理: {len(successful_tasks)} 个目录")
    if successful_tasks:
        for name in successful_tasks:
            print(f"    - ✅ {name}")
    print(f"\n  失败或跳过: {len(failed_tasks)} 个目录")
    if failed_tasks:
        for name in failed_tasks:
            print(f"    - ❌ {name}")
    print("="*60)


if __name__ == "__main__":
    main()