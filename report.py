import argparse
import importlib
import os
import warnings
import time
from datetime import timedelta

from soi.local import EnvironmentSettings

def setup_parser():
    """
    设置命令行参数解析器。
    允许用户选择追踪器、数据集、mask 轮次等。
    """
    parser = argparse.ArgumentParser(description="运行目标跟踪器报告生成")
    parser.add_argument("--tracker", type=str, default="odtrack", help="选择追踪器名称")
    parser.add_argument("--dataset", type=str, default="lasot", choices=["lasot", "otb", "vot", "videocube"], help="选择数据集名称")
    parser.add_argument("--subsets", type=str, default="test", help="子数据集名称")
    parser.add_argument("--save_dir", type=str, default="nips25/", help="结果保存路径")
    parser.add_argument("--masked_report", action="store_true", help="是否测试 masked 后的序列跟踪性能")
    parser.add_argument("--mask_round", type=int, default=1, help="当前 mask 轮次编号，用于区分不同的评估报告")

    return parser

def main():
    """
    追踪器报告生成入口。
    直接初始化 `experiment` 并调用 `report_masked()`。
    """
    warnings.filterwarnings("ignore")
    parser = setup_parser()
    args = parser.parse_args()
    
    # 计算保存目录
    save_dir = os.path.join(args.save_dir, args.dataset, args.subsets, args.tracker)
    
    # 加载实验环境
    evs = EnvironmentSettings()
    dataset_dir = evs.find_root_dir(dataset_str=args.dataset)

    # 加载实验模块
    expr_module = importlib.import_module(f"soi.experiments.{args.dataset}")
    expr_func = getattr(expr_module, "call_back")
    exper_class = expr_func()

    # 初始化 `experiment`
    if args.dataset in ["vot", "votSOI"]:
        experiment = exper_class(dataset_dir, save_dir, version=2019)
    else:
        experiment = exper_class(dataset_dir, save_dir, args.subsets)

    print("📊 启动报告生成...")
    start_time = time.time()
    
    tracker_names = [args.tracker]
    if args.masked_report:
        experiment.report_masked(tracker_names, mask_round=args.mask_round)
    elif args.dataset == "videocubeSOI":
        tracker_names = [name + "_restart" for name in tracker_names]
        experiment.report(tracker_names)
        experiment.report_robust(tracker_names)
    else:
        experiment.report(tracker_names)

    total_time = timedelta(seconds=(time.time() - start_time))
    print(f"✅ 报告生成完成，总耗时: {total_time}")

if __name__ == "__main__":
    main()


# python report.py --tracker odtrack --dataset lasot --subsets test 
# python report.py --tracker odtrack --dataset lasot --subsets test  --masked_report --mask_round 2
