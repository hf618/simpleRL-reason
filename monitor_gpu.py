# -*- coding: utf-8 -*-
import subprocess
import time
import datetime
import argparse
import sys
import os

def get_gpu_info(selected_indices=None):
    """
    通过 nvidia-smi 命令获取 GPU 上的所有进程和利用率信息。
    【最终版方法】此版本解析 nvidia-smi 的默认文本输出，以确保捕获所有可见进程，
    解决了查询 API 在某些系统上可能失效的问题。
    """
    processes = []
    gpu_utilizations = {}

    # 1. 查询每个 GPU 的总体利用率 (这部分方法可靠，保持不变)
    try:
        util_query = "index,utilization.gpu"
        cmd_util = f"nvidia-smi --query-gpu={util_query} --format=csv,noheader,nounits"
        util_info_str = subprocess.check_output(cmd_util, shell=True, text=True, encoding='utf-8')
        gpu_utilizations = {line.split(', ')[0]: line.split(', ')[1] for line in util_info_str.strip().splitlines()}
    except subprocess.CalledProcessError as e:
        print(f"\n[警告] 无法查询 GPU 利用率，错误: {e.stderr.strip()}", file=sys.stderr)
        gpu_utilizations = {}

    # 2. 【核心修改】运行原生 nvidia-smi 命令并解析其输出以获取进程列表
    try:
        smi_output = subprocess.check_output("nvidia-smi", shell=True, text=True, encoding='utf-8', stderr=subprocess.PIPE)
        lines = smi_output.splitlines()

        in_processes_section = False
        process_header_passed = False
        
        for line in lines:
            # 定位到进程信息表的开始
            if line.startswith("| Processes:"):
                in_processes_section = True
                continue

            if not in_processes_section:
                continue
            
            # 跳过表头和分隔线
            if line.startswith("|  GPU   GI   CI"):
                process_header_passed = True
                continue
            
            # 如果还没到表头，或者已经过了进程表，就跳过
            if not process_header_passed or line.startswith("+--"):
                continue

            # 解析实际的进程行
            if line.startswith("|"):
                try:
                    parts = line.strip()[1:-1].strip().split()
                    # 一个有效的进程行至少有7个部分 (GPU, GI, CI, PID, Type, Name, Memory)
                    if len(parts) < 7:
                        continue

                    gpu_index = parts[0]
                    pid = parts[3]
                    # 进程名可能包含空格，从第6个元素到倒数第二个元素都是进程名
                    process_name = " ".join(parts[5:-1])
                    memory_used = parts[-1].replace('MiB', '')

                    # 确保 PID 是有效的数字，并且进程名不为空
                    if pid.isdigit() and process_name:
                        processes.append([gpu_index, pid, process_name, memory_used])

                except (IndexError, ValueError) as parse_error:
                    # 如果某行解析失败，打印一个警告并继续
                    print(f"\n[警告] 解析行失败: '{line}', 错误: {parse_error}", file=sys.stderr)
                    continue
    
    except subprocess.CalledProcessError as e:
        print(f"\n[严重错误] 执行 'nvidia-smi' 失败: {e.stderr.strip()}", file=sys.stderr)
        return [], gpu_utilizations # 发生严重错误，返回空列表

    # 3. 如果用户指定了 GPU 索引，则进行筛选 (这部分保持不变)
    if selected_indices:
        selected_indices_str = [str(i) for i in selected_indices]
        filtered_processes = [p for p in processes if p[0] in selected_indices_str]
        filtered_gpu_utils = {idx: util for idx, util in gpu_utilizations.items() if idx in selected_indices_str}
        return filtered_processes, filtered_gpu_utils

    return processes, gpu_utilizations

def main():
    """
    主函数，用于解析参数和执行监控循环。
    """
    parser = argparse.ArgumentParser(
        description="NVIDIA GPU 进程监控脚本。在指定时间内，周期性记录 GPU 进程信息到日志文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-H', '--hours',
        type=float,
        required=True,
        help="需要监控的总时长（单位：小时）。例如：1.5 代表一个半小时。"
    )
    parser.add_argument(
        '-S', '--seconds',
        type=int,
        required=True,
        help="每次采集数据的时间间隔（单位：秒）。例如：10"
    )
    parser.add_argument(
        '-g', '--gpu-indices',
        type=int,
        nargs='+',
        default=None,
        help="要监控的 GPU 索引列表，用空格隔开。\n例如：-g 0 1 3。如果未提供，则监控所有可用的 GPU。"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help="输出路径（文件或文件夹）。\n"
             "如果提供文件夹路径 (如 'logs/' 或 'logs')，将在该文件夹内自动生成带时间戳的文件名。\n"
             "如果提供完整文件名 (如 'logs/my_log.csv')，将直接使用该名称。\n"
             "如果未提供，将在当前目录生成带时间戳的文件名。"
    )
    args = parser.parse_args()

    # --- 文件名处理逻辑 ---
    output_path = args.output
    if output_path is None:
        # Case 1: 未提供输出路径，在当前目录生成带时间戳的文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # 【已修改】移除 "gpu_monitor_log_" 前缀
        log_filename = f'{timestamp}.csv'
    else:
        # 检查提供的路径是文件夹还是文件
        if output_path.endswith(('/', '\\')) or ('.' not in os.path.basename(output_path) and os.path.splitext(output_path)[1] == ''):
            # Case 2: 提供了文件夹路径
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            # 【已修改】移除 "gpu_monitor_log_" 前缀
            log_filename = os.path.join(output_path, f'{timestamp}.csv')
        else:
            # Case 3: 提供了完整的文件路径
            log_filename = output_path
            parent_dir = os.path.dirname(log_filename)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    monitor_duration_seconds = args.hours * 3600
    sample_interval_seconds = args.seconds

    try:
        subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：无法执行 'nvidia-smi' 命令。")
        print("请确保 NVIDIA 驱动已正确安装并且 'nvidia-smi' 在您的系统 PATH 中。")
        sys.exit(1)

    # 准备要在控制台和日志文件中显示的信息
    gpu_monitored_str = str(args.gpu_indices) if args.gpu_indices is not None else '所有'
    log_filename_abs = os.path.abspath(log_filename)

    # 在控制台打印启动信息
    print(f"开始监控 GPU 使用情况...")
    print(f"总时长: {args.hours} 小时")
    print(f"采样间隔: {args.seconds} 秒")
    print(f"监控的 GPU 索引: {gpu_monitored_str}")
    print(f"日志文件: {log_filename_abs}")
    print("-" * 30)

    start_time = time.time()
    end_time = start_time + monitor_duration_seconds

    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            # 1. 【新增】写入标题信息到日志文件
            title_block = (
                f"# 开始监控 GPU 使用情况...\n"
                f"# 总时长: {args.hours} 小时\n"
                f"# 采样间隔: {args.seconds} 秒\n"
                f"# 监控的 GPU 索引: {gpu_monitored_str}\n"
                f"# 日志文件: {log_filename_abs}\n"
                f"#{'-'*60}\n\n"
            )
            f.write(title_block)

            # 2. 写入 CSV 表头
            header = "Timestamp,GPU_Index,PID,Process_Name,Memory_Used_MiB,GPU_Util_Percent\n"
            f.write(header)
            f.flush()

            while time.time() < end_time:
                current_loop_start = time.time()
                
                now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                processes, gpu_utils = get_gpu_info(args.gpu_indices)

                if not processes and not gpu_utils:
                    print(f"[{now_str}] 未找到指定的 GPU 或无法获取其信息。请检查 GPU 索引是否正确。")
                elif not processes:
                    print(f"[{now_str}] 未在指定的 GPU 上检测到任何进程。仅记录总体利用率。")
                    for gpu_index, util in gpu_utils.items():
                         log_line = f"{now_str},{gpu_index},N/A,N/A,0,{util}\n"
                         f.write(log_line)
                else:
                    print(f"[{now_str}] 成功记录 {len(processes)} 个进程信息。")
                    for proc in processes:
                        gpu_index, pid, proc_name, memory_used = proc
                        util = gpu_utils.get(gpu_index, 'N/A') 
                        log_line = f"{now_str},{gpu_index},{pid},{proc_name},{memory_used},{util}\n"
                        f.write(log_line)
                
                # 3. 【新增】写入记录间隔符
                f.write("-" * 80 + "\n")
                f.flush()

                elapsed_time = time.time() - current_loop_start
                sleep_time = sample_interval_seconds - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n监控被用户手动中断。")
    except Exception as e:
        print(f"\n发生严重错误: {e}")
    finally:
        print(f"监控结束。数据已保存至 {log_filename}")

if __name__ == "__main__":
    main()
