from __future__ import absolute_import
import argparse
import importlib
import os
import warnings
import multiprocessing
import time
from datetime import timedelta
import torch.multiprocessing as mp
import torch

from soi.trackers import TrackerODTrack
from soi.local import EnvironmentSettings
from soi.utils.analyze_soi_ratios import analyze_retio


import sys
import logging


def setup_logger(worker_id=None, single_thread=False):
    """ 
    创建 logger：
    - 单线程模式 (`MainProcess`) → `logs/main.log`
    - 多线程模式 (`Worker-X`) → `logs/worker_X.log`
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 确保 logs 目录存在

    if single_thread:
        log_file = os.path.join(log_dir, "main.log")
        logger_name = "main_logger"
    else:
        log_file = os.path.join(log_dir, f"worker_{worker_id}.log")
        logger_name = f"worker_{worker_id}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 确保 handler 只被添加一次，避免重复日志
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s [THREAD-%(name)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def run_single_sequence_wrapper(experiment, seq_idx, tracker, kwargs):
    """ 运行单个任务（在多线程模式下，每个线程写入同一个日志文件） """
    worker_name = multiprocessing.current_process().name
    single_thread = worker_name == "MainProcess"

    # **获取 worker_id**
    if single_thread:
        worker_id = 0
    else:
        try:
            worker_id = int(worker_name.split("-")[-1]) - 1  # **获取进程编号**
        except ValueError:
            worker_id = 0

    # **初始化 Logger**
    logger = setup_logger(worker_id, single_thread=single_thread)
    logger.info(f"线程 {worker_id} 任务 {seq_idx} 开始")

    # **多线程模式下，重定向 stdout/stderr 到日志文件**
    if not single_thread:
        sys.stdout = open(f"logs/worker_{worker_id}_stdout.log", "w")
        sys.stderr = sys.stdout

    try:
        num_gpu = torch.cuda.device_count()  # 获取 GPU 数量
        gpu_id = worker_id % num_gpu if num_gpu > 0 else 0  # **防止 GPU 数量为 0**

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
        logger.info(f"[Process {worker_name}] 分配到 GPU: {gpu_id}")

        # **运行单个序列的实验，并传递 `logger`**
        experiment.run_single_sequence(seq_idx, tracker, logger=logger, **kwargs)
        logger.info(f"线程 {worker_id} 任务 {seq_idx} 完成")
    except Exception as e:
        logger.error(f"线程 {worker_id} 任务 {seq_idx} 失败: {e}")
    finally:
        if not single_thread:
            sys.stdout.close()  # 关闭日志文件


def choose_tracker(name):
    """
    根据提供的名称选择适当的追踪器类。
    """
    tracker_mapping = {
        "odtrack": TrackerODTrack,
    }
    return tracker_mapping.get(name)

def get_list_file(dataset, save_dir, subsets):
    """
    获取数据集的列表文件路径。
    """
    if dataset == "got10kSOI":
        return os.path.join(save_dir, "../data/got10k", subsets, "all-list.txt")
    else:
        return None

def setup_parser():
    """
    设置命令行参数解析器。
    允许用户选择追踪器、数据集、运行模式等。
    """
    parser = argparse.ArgumentParser(description="运行目标跟踪器，支持多种数据集和设置。")
    parser.add_argument("--tracker", type=str, default="odtrack", help="选择追踪器名称")
    parser.add_argument("--dataset", type=str, default="lasot", choices=["lasot", "otb", "vot", "videocube"], help="选择数据集名称")
    parser.add_argument("--subsets", type=str, default="test", help="子数据集名称")

    parser.add_argument("--save_dir", type=str, default="nips25/", help="结果保存路径")
    
    parser.add_argument("--cuda", type=str, default="0", help="CUDA设备编号")

    parser.add_argument("--run_soi", action="store_true", help="是否启用SOI统计")
    parser.add_argument("--visual", action="store_true", help="是否可视化结果")
    parser.add_argument("--run_mask", action="store_true", help="是否运行mask模式")

    parser.add_argument("--report", action="store_true", help="是否生成报告")
    parser.add_argument("--masked_report", action="store_true", help="测试masked后的序列跟踪性能")
    
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single", help="运行模式：单线程(single)或多线程(multi)")
    parser.add_argument("--threshold", type=float, default=0.25, help="是否使用SOI阈值参数")
    parser.add_argument("--num_threads", type=int, default=2, help="多线程模式下的线程数")
    return parser

def main():
    """
    追踪器的主入口函数。
    根据用户输入选择单线程或多线程模式进行跟踪实验。
    """
    warnings.filterwarnings("ignore")
    parser = setup_parser()
    args = parser.parse_args()
    
    save_dir = os.path.join(args.save_dir, args.dataset, args.subsets, args.tracker)  # nips25/lasot/test/odtrack
    evs = EnvironmentSettings()
    dataset_dir = evs.find_root_dir(dataset_str=args.dataset)
    list_file = get_list_file(args.dataset, args.save_dir, args.subsets)
    
    # 加载数据集对应的实验模块
    expr_module = importlib.import_module(f"soi.experiments.{args.dataset}")
    expr_func = getattr(expr_module, "call_back")
    exper_class = expr_func()
    
    # 初始化实验类
    if args.dataset in ["vot", "votSOI"]:
        experiment = exper_class(dataset_dir, save_dir, version=2019)
    else:
        experiment = exper_class(dataset_dir, save_dir, args.subsets, list_file=list_file)
    
    print("启动追踪器运行...")
    start_time = time.time()
    
    if args.mode == "single":
        print("运行模式：单线程")
        for seq_idx in range(len(experiment.dataset)):
            run_single_sequence_wrapper(experiment, seq_idx, args.tracker, {
                "save_dir": save_dir,
                "visualize": args.visual,
                "run_mask": args.run_mask,
                "mask_info": getattr(args, "mask_info", None),
                "run_soi": args.run_soi,
                "threshold": args.threshold,
            })
    else:
        print("运行模式：多线程")
        multiprocessing.set_start_method("spawn", force=True)  # 确保 spawn 方式初始化
        
        num_threads = args.num_threads if args.num_threads else multiprocessing.cpu_count()
        trackers = [args.tracker]

        param_list = [
            (experiment, seq_idx, tracker, {
                "save_dir": save_dir,
                "visualize": args.visual,
                "run_mask": args.run_mask,
                "mask_info": getattr(args, "mask_info", None),
                "run_soi": args.run_soi,
                "threshold": args.threshold,
            })
            for seq_idx in range(len(experiment.dataset))
            for tracker in trackers
        ]

        print(f"共生成 {len(param_list)} 个任务，使用 {num_threads} 个线程进行并行处理。")
        
        with mp.Pool(processes=num_threads) as pool:
            try:
                pool.starmap(run_single_sequence_wrapper, param_list)
                pool.close()  # 关闭进程池，防止新的任务提交
                pool.join()   # 等待所有子进程完成
            except KeyboardInterrupt:
                print("检测到用户终止 (Ctrl+C)，正在清理进程...")
                pool.terminate()  # 强制终止所有进程
                pool.join()  # 确保所有进程完全关闭
            except Exception as e:
                print(f"[Error] 进程池运行时发生异常: {e}")
                pool.terminate()
                pool.join()
    
    total_time = timedelta(seconds=(time.time() - start_time))
    print(f"追踪完成，总耗时: {total_time}")
    
    if args.report:
        tracker_names = [args.tracker]
        if args.masked_report:
            experiment.report_masked(tracker_names)
        elif args.dataset == "videocubeSOI":
            tracker_names = [name + "_restart" for name in tracker_names]
            experiment.report(tracker_names, args.report_th)
            experiment.report_robust(tracker_names)
        else:
            experiment.report(tracker_names)
        print("报告生成完成")

if __name__ == "__main__":
    main()


# python run_dist.py --run_soi --mode multi --num_threads 3