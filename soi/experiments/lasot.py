from __future__ import absolute_import

import os
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# from experiments.otb import ExperimentOTB
from soi.datasets.lasot import LaSOT
from soi.utils.metrics import rect_iou, center_error, normalized_center_error
from soi.utils.screening_util import *
from soi.utils.help import makedir
from soi.trackers import TrackerODTrack


def choose_tracker(name):
    """根据名称选择追踪器类"""
    tracker_mapping = {
        # 'keeptrack': TrackerKeepTrack,
        # 'ostrack': TrackerOSTrack,
        "odtrack": TrackerODTrack,
    }
    return tracker_mapping.get(name)


def call_back():
    return ExperimentLaSOT


class ExperimentLaSOT(object):
    r"""Experiment pipeline and evaluation toolkit for LaSOT dataset.

    Args:
        root_dir (string): Root directory of LaSOT dataset.
        subset (string, optional): Specify ``train`` or ``test``
            subset of LaSOT.  Default is ``test``.
        return_meta (bool, optional): whether to fetch meta info
        (occlusion or out-of-view).  Default is ``False``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(
        self, root_dir, save_dir, subset, return_meta=False, th=0.1, list_file=None
    ):
        # assert subset.upper() in ['TRAIN', 'TEST']
        self.root_dir = root_dir
        self.subset = subset
        self.dataset = LaSOT(root_dir, subset, return_meta=return_meta)

        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_nce = 51
        """
        -save_dir
            |--TRACKER_NAME
                |---DATASET
                    |--result
                    |--reports
                    |--image
                    |--scores
        """
        self.result_dir = os.path.join(save_dir, "results")
        self.report_dir = os.path.join(save_dir, "reports")
        self.mask_info_dir = os.path.join(save_dir, "mask_info")
        # self.score_map_dir = os.path.join(save_dir, "score_map")
        self.result_dir_masked = os.path.join(save_dir, "results_masked")
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.mask_info_dir)
        # makedir(self.score_map_dir)
        makedir(self.result_dir_masked)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, "performance.json")

        performance = {}
        for name in tracker_names:
            print("Evaluating", name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            speeds = np.zeros(seq_num)

            performance.update({name: {"overall": {}, "seq_wise": {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                # result_dir = os.path.join(self.result_dir[:self.result_dir.find('SOI') + 3], 'tracker_result', name,
                #                           self.dataset.name,
                #                           self.result_dir[self.result_dir.find(self.dataset.subset[0]):])
                # record_file = os.path.join(
                #     result_dir, name, '%s.txt' % seq_name)
                record_file = os.path.join(self.result_dir, "%s.txt" % seq_name)
                boxes = np.loadtxt(record_file, delimiter=",")
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    # from IPython import embed;embed()
                    print("warning: %s anno donnot match boxes" % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors, norm_center_errors = self._calc_metrics(
                    boxes, anno
                )
                succ_curve[s], prec_curve[s], norm_prec_curve[s] = self._calc_curves(
                    ious, center_errors, norm_center_errors
                )

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, "times/%s_time.txt" % seq_name
                )
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1.0 / times)

                # store sequence-wise performance
                performance[name]["seq_wise"].update(
                    {
                        seq_name: {
                            "success_curve": succ_curve[s].tolist(),
                            "precision_curve": prec_curve[s].tolist(),
                            "normalized_precision_curve": norm_prec_curve[s].tolist(),
                            "success_score": np.mean(succ_curve[s]),
                            "precision_score": prec_curve[s][20],
                            "normalized_precision_score": np.mean(norm_prec_curve[s]),
                            "success_rate": succ_curve[s][self.nbins_iou // 2],
                            "speed_fps": speeds[s] if speeds[s] > 0 else -1,
                        }
                    }
                )

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            norm_prec_curve = np.mean(norm_prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            norm_prec_score = np.mean(norm_prec_curve)
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]["overall"].update(
                {
                    "success_curve": succ_curve.tolist(),
                    "precision_curve": prec_curve.tolist(),
                    "normalized_precision_curve": norm_prec_curve.tolist(),
                    "success_score": succ_score,
                    "precision_score": prec_score,
                    "normalized_precision_score": norm_prec_score,
                    "success_rate": succ_rate,
                    "speed_fps": avg_speed,
                }
            )

        # report the performance
        with open(report_file, "w") as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        self.plot_curves(tracker_names)

        return performance

    def report_masked(self, tracker_names, mask_round=1):
        """
        评估 mask 处理后的跟踪器性能，并保存 JSON 报告。

        参数：
            tracker_names (list): 参与评估的跟踪器名称列表。
            mask_round (int): 当前是第几轮 mask 评估，用于区分不同轮次的结果。

        返回：
            performance (dict): 记录所有评估指标的字典。
        """
        assert isinstance(tracker_names, (list, tuple))

        # 创建带有 mask 轮次的报告目录
        report_dir = os.path.join(self.report_dir, f"{tracker_names[0]}_mask_round_{mask_round}")
        os.makedirs(report_dir, exist_ok=True)

        # 生成带有 mask 轮次的 JSON 文件
        report_file = os.path.join(report_dir, f"performance_mask_round_{mask_round}.json")

        performance = {}
        for name in tracker_names:
            print(f"Evaluating {name} (Mask Round {mask_round})")
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            norm_prec_curve = np.zeros((seq_num, self.nbins_nce))
            speeds = np.zeros(seq_num)

            performance.update({name: {"overall": {}, "seq_wise": {}, "mask_round": mask_round}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(self.result_dir.replace('results', 'results_masked'), f"{seq_name}.txt")

                if not os.path.exists(record_file):
                    print(f"⚠️ Warning: {seq_name} tracking result missing for {name}, skipping.")
                    continue

                boxes = np.loadtxt(record_file, delimiter=",")
                boxes[0] = anno[0]

                if len(boxes) != len(anno):
                    print(f"⚠️ Warning: {seq_name} annotation does not match boxes for {name}. Adjusting size.")
                    len_min = min(len(boxes), len(anno))
                    boxes, anno = boxes[:len_min], anno[:len_min]

                assert len(boxes) == len(anno)

                ious, center_errors, norm_center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], prec_curve[s], norm_prec_curve[s] = self._calc_curves(ious, center_errors, norm_center_errors)

                # 计算平均跟踪速度
                time_file = os.path.join(self.result_dir, name, f"times/{seq_name}_time.txt")
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    speeds[s] = np.mean(1.0 / times) if len(times) > 0 else -1

                # 存储单个序列的结果
                performance[name]["seq_wise"][seq_name] = {
                    "success_curve": succ_curve[s].tolist(),
                    "precision_curve": prec_curve[s].tolist(),
                    "normalized_precision_curve": norm_prec_curve[s].tolist(),
                    "success_score": np.mean(succ_curve[s]),
                    "precision_score": prec_curve[s][20],
                    "normalized_precision_score": np.mean(norm_prec_curve[s]),
                    "success_rate": succ_curve[s][self.nbins_iou // 2],
                    "speed_fps": speeds[s] if speeds[s] > 0 else -1,
                }

            # 计算整体性能指标
            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            norm_prec_curve = np.mean(norm_prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            norm_prec_score = np.mean(norm_prec_curve)
            succ_rate = succ_curve[self.nbins_iou // 2]
            avg_speed = np.mean(speeds[speeds > 0]) if np.count_nonzero(speeds) > 0 else -1

            # 存储整体性能结果
            performance[name]["overall"] = {
                "success_curve": succ_curve.tolist(),
                "precision_curve": prec_curve.tolist(),
                "normalized_precision_curve": norm_prec_curve.tolist(),
                "success_score": succ_score,
                "precision_score": prec_score,
                "normalized_precision_score": norm_prec_score,
                "success_rate": succ_rate,
                "speed_fps": avg_speed,
            }

        # 保存 JSON 报告
        with open(report_file, "w") as f:
            json.dump(performance, f, indent=4)

        # 绘制曲线
        self.plot_curves(tracker_names, mask_round=mask_round)

        return performance

    def _calc_metrics(self, boxes, anno):
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print("Warning: no valid annotations")
            return None, None, None
        else:
            ious = rect_iou(boxes[valid, :], anno[valid, :])
            center_errors = center_error(boxes[valid, :], anno[valid, :])
            norm_center_errors = normalized_center_error(
                boxes[valid, :], anno[valid, :]
            )
            return ious, center_errors, norm_center_errors

    def _calc_curves(self, ious, center_errors, norm_center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        thr_nce = np.linspace(0, 0.5, self.nbins_nce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)
        bin_nce = np.less_equal(norm_center_errors, thr_nce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)
        norm_prec_curve = np.mean(bin_nce, axis=0)

        return succ_curve, prec_curve, norm_prec_curve

    # def plot_curves(self, tracker_names, extension=".png"):
    #     # assume tracker_names[0] is your tracker
    #     report_dir = os.path.join(self.report_dir, tracker_names[0])
    #     assert os.path.exists(report_dir), (
    #         'No reports found. Run "report" first' "before plotting curves."
    #     )
    #     report_file = os.path.join(report_dir, "performance.json")
    #     assert os.path.exists(report_file), (
    #         'No reports found. Run "report" first' "before plotting curves."
    #     )

    #     # load pre-computed performance
    #     with open(report_file) as f:
    #         performance = json.load(f)

    #     succ_file = os.path.join(report_dir, "success_plots" + extension)
    #     prec_file = os.path.join(report_dir, "precision_plots" + extension)
    #     norm_prec_file = os.path.join(report_dir, "norm_precision_plots" + extension)
    #     key = "overall"

    #     # markers
    #     markers = ["-", "--", "-."]
    #     markers = [c + m for m in markers for c in [""] * 10]

    #     # filter performance by tracker_names
    #     performance = {k: v for k, v in performance.items() if k in tracker_names}

    #     # sort trackers by success score
    #     tracker_names = list(performance.keys())
    #     succ = [t[key]["success_score"] for t in performance.values()]
    #     inds = np.argsort(succ)[::-1]
    #     tracker_names = [tracker_names[i] for i in inds]

    #     # plot success curves
    #     thr_iou = np.linspace(0, 1, self.nbins_iou)
    #     fig, ax = plt.subplots()
    #     lines = []
    #     legends = []
    #     for i, name in enumerate(tracker_names):
    #         (line,) = ax.plot(
    #             thr_iou,
    #             performance[name][key]["success_curve"],
    #             markers[i % len(markers)],
    #         )
    #         lines.append(line)
    #         legends.append(
    #             "%s: [%.3f]" % (name, performance[name][key]["success_score"])
    #         )
    #     matplotlib.rcParams.update({"font.size": 7.4})
    #     # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
    #     legend = ax.legend(lines, legends, loc="lower left", bbox_to_anchor=(0.0, 0.0))

    #     matplotlib.rcParams.update({"font.size": 9})
    #     ax.set(
    #         xlabel="Overlap threshold",
    #         ylabel="Success rate",
    #         xlim=(0, 1),
    #         ylim=(0, 1),
    #         title="Success plots on LaSOT",
    #     )
    #     ax.grid(True)
    #     fig.tight_layout()

    #     # control ratio
    #     # ax.set_aspect('equal', 'box')

    #     print("Saving success plots to", succ_file)
    #     fig.savefig(
    #         succ_file, bbox_extra_artists=(legend,), bbox_inches="tight", dpi=300
    #     )

    #     # sort trackers by precision score
    #     tracker_names = list(performance.keys())
    #     prec = [t[key]["precision_score"] for t in performance.values()]
    #     inds = np.argsort(prec)[::-1]
    #     tracker_names = [tracker_names[i] for i in inds]

    #     # plot precision curves
    #     thr_ce = np.arange(0, self.nbins_ce)
    #     fig, ax = plt.subplots()
    #     lines = []
    #     legends = []
    #     for i, name in enumerate(tracker_names):
    #         (line,) = ax.plot(
    #             thr_ce,
    #             performance[name][key]["precision_curve"],
    #             markers[i % len(markers)],
    #         )
    #         lines.append(line)
    #         legends.append(
    #             "%s: [%.3f]" % (name, performance[name][key]["precision_score"])
    #         )
    #     matplotlib.rcParams.update({"font.size": 7.4})
    #     # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
    #     legend = ax.legend(lines, legends, loc="lower right", bbox_to_anchor=(1.0, 0.0))

    #     matplotlib.rcParams.update({"font.size": 9})
    #     ax.set(
    #         xlabel="Location error threshold",
    #         ylabel="Precision",
    #         xlim=(0, thr_ce.max()),
    #         ylim=(0, 1),
    #         title="Precision plots on LaSOT",
    #     )
    #     ax.grid(True)
    #     fig.tight_layout()

    #     # control ratio
    #     # ax.set_aspect('equal', 'box')

    #     print("Saving precision plots to", prec_file)
    #     fig.savefig(prec_file, dpi=300)

    #     # added by user
    #     # sort trackers by normalized precision score
    #     tracker_names = list(performance.keys())
    #     prec = [t[key]["normalized_precision_score"] for t in performance.values()]
    #     inds = np.argsort(prec)[::-1]
    #     tracker_names = [tracker_names[i] for i in inds]

    #     # plot normalized precision curves
    #     thr_nce = np.arange(0, self.nbins_nce)
    #     fig, ax = plt.subplots()
    #     lines = []
    #     legends = []
    #     for i, name in enumerate(tracker_names):
    #         (line,) = ax.plot(
    #             thr_nce,
    #             performance[name][key]["normalized_precision_curve"],
    #             markers[i % len(markers)],
    #         )
    #         lines.append(line)
    #         legends.append(
    #             "%s: [%.3f]"
    #             % (name, performance[name][key]["normalized_precision_score"])
    #         )
    #     matplotlib.rcParams.update({"font.size": 7.4})
    #     # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
    #     legend = ax.legend(lines, legends, loc="lower right", bbox_to_anchor=(1.0, 0.0))

    #     matplotlib.rcParams.update({"font.size": 9})
    #     ax.set(
    #         xlabel="Normalized location error threshold",
    #         ylabel="Normalized precision",
    #         xlim=(0, thr_ce.max()),
    #         ylim=(0, 1),
    #         title="Normalized precision plots on LaSOT",
    #     )
    #     ax.grid(True)
    #     fig.tight_layout()

    #     # control ratio
    #     # ax.set_aspect('equal', 'box')

    #     print("Saving normalized precision plots to", norm_prec_file)
    #     fig.savefig(norm_prec_file, dpi=300)

    def plot_curves(self, tracker_names, mask_round=None, extension=".png"):
        """ 绘制 success、precision、normalized precision 曲线，支持原始 & mask 结果区分 """
        
        # 选择不同的报告目录
        if mask_round is None:
            report_dir = os.path.join(self.report_dir, tracker_names[0])  # 原始结果
            report_file = os.path.join(report_dir, "performance.json")
            title_suffix = "Original"
        else:
            report_dir = os.path.join(self.report_dir, f"{tracker_names[0]}_mask_round_{mask_round}")  # mask 结果
            report_file = os.path.join(report_dir, f"performance_mask_round_{mask_round}.json")
            title_suffix = f"Mask {mask_round}"
        
        assert os.path.exists(report_dir), 'No reports found. Run "report" before plotting curves.'
        assert os.path.exists(report_file), 'No performance data found, run "report" first.'
        
        # 读取 JSON 结果
        with open(report_file) as f: performance = json.load(f)
        
        # 生成文件名，确保 mask 结果不会覆盖原始结果
        suffix = f"_mask_{mask_round}" if mask_round is not None else "_original"
        succ_file, prec_file, norm_prec_file = [os.path.join(report_dir, f"{t}_plot{suffix}{extension}") for t in ["success", "precision", "norm_precision"]]
        
        key, markers = "overall", ["-", "--", "-.", ":"] * 10  # 预设 line markers
        
        # 过滤 tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}
        tracker_names = sorted(performance.keys(), key=lambda n: performance[n][key]["success_score"], reverse=True)

        # 绘制 success 曲线
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            (line,) = ax.plot(np.linspace(0, 1, self.nbins_iou), performance[name][key]["success_curve"], markers[i % len(markers)])
            lines.append(line)
            legends.append(f"{name}: [{performance[name][key]['success_score']:.3f}]")
        ax.legend(lines, legends, loc="lower left", bbox_to_anchor=(0.0, 0.0))
        ax.set(xlabel="Overlap threshold", ylabel="Success rate", xlim=(0, 1), ylim=(0, 1), title=f"Success plots on {self.dataset.name} ({title_suffix})")
        ax.grid(True)
        fig.tight_layout()
        print(f"Saving success plots to {succ_file}")
        fig.savefig(succ_file, dpi=300)

        # 绘制 precision 曲线
        tracker_names = sorted(performance.keys(), key=lambda n: performance[n][key]["precision_score"], reverse=True)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            (line,) = ax.plot(np.arange(0, self.nbins_ce), performance[name][key]["precision_curve"], markers[i % len(markers)])
            lines.append(line)
            legends.append(f"{name}: [{performance[name][key]['precision_score']:.3f}]")
        ax.legend(lines, legends, loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax.set(xlabel="Location error threshold", ylabel="Precision", xlim=(0, self.nbins_ce - 1), ylim=(0, 1), title=f"Precision plots on {self.dataset.name} ({title_suffix})")
        ax.grid(True)
        fig.tight_layout()
        print(f"Saving precision plots to {prec_file}")
        fig.savefig(prec_file, dpi=300)

        # 绘制 normalized precision 曲线
        tracker_names = sorted(performance.keys(), key=lambda n: performance[n][key]["normalized_precision_score"], reverse=True)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            (line,) = ax.plot(np.arange(0, self.nbins_nce), performance[name][key]["normalized_precision_curve"], markers[i % len(markers)])
            lines.append(line)
            legends.append(f"{name}: [{performance[name][key]['normalized_precision_score']:.3f}]")
        ax.legend(lines, legends, loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax.set(xlabel="Normalized location error threshold", ylabel="Normalized precision", xlim=(0, self.nbins_nce - 1), ylim=(0, 1), title=f"Normalized Precision plots on {self.dataset.name} ({title_suffix})")
        ax.grid(True)
        fig.tight_layout()
        print(f"Saving normalized precision plots to {norm_prec_file}")
        fig.savefig(norm_prec_file, dpi=300)


    def run(self, tracker, visualize=False):
        print(
            "Running tracker %s on %s..." % (tracker.name, type(self.dataset).__name__)
        )

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print("--Sequence %d/%d: %s" % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, "%s.txt" % seq_name
            )
            if os.path.exists(record_file):
                print("  Found results, skipping", seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(seq_name, img_files, anno, visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)

    # def run_single_sequence(self, seq_index, tracker, **kwargs):
    #     """
    #     在单独的进程中运行单个序列的跟踪任务。

    #     Args:
    #         seq_index: 数据集序列的索引。
    #         tracker: 跟踪器实例。
    #         **kwargs: 可选参数集合，例如：
    #                   - save_dir: 路径
    #                   - visualize: 是否进行可视化
    #                   - run_mask: 是否使用遮罩模式
    #                   - mask_info: 遮罩信息
    #                   - use_filter: 是否使用 `track_and_filter_candidates`
    #                   - threshold: track_and_filter_candidates 的阈值
    #     """
    #     # 初始化 tracker
    #     tracker_cls = choose_tracker(tracker)
    #     tracker = tracker_cls()

    #     seq_name = self.dataset.seq_names[seq_index]
    #     img_files, anno = self.dataset[seq_index]

    #     print(f"Processing Sequence {seq_index + 1}/{len(self.dataset)}: {seq_name}...")

    #     # 提取关键字参数
    #     save_dir = kwargs.get("save_dir", False)
    #     visualize = kwargs.get("visualize", False)
    #     run_mask = kwargs.get("run_mask", False)
    #     run_soi = kwargs.get("run_soi", False)
    #     threshold = kwargs.get("threshold", 0.25)  # 默认阈值为0.25
    #     # 构建结果保存路径
    #     result_dir = self.result_dir_masked if run_mask else self.result_dir
    #     record_file = os.path.join(result_dir, f"{seq_name}.txt")
    #     if os.path.exists(record_file):
    #         print(f"  Found results, skipping {seq_name}")
    #         return
        
    #     mask_info = None
    #     if run_mask:
    #         # 计算 mask 数据存储目录
    #         masked_base_dir = f"{save_dir}/../{tracker.name}/results/"
    #         mask_jsonl_path = os.path.join(masked_base_dir, seq_name, "masked_info.jsonl")

    #         if os.path.exists(mask_jsonl_path):  # 如果 JSONL 文件存在
    #             mask_info = []  # 初始化 mask 信息存储列表
    #             with open(mask_jsonl_path, "r") as f:
    #                 for line in f:  # 逐行读取 JSONL 文件
    #                     try:
    #                         mask_info.append(json.loads(line.strip()))  # 解析 JSON 并添加到列表
    #                     except json.JSONDecodeError:
    #                         print(f"  [Error] Invalid JSON line in {mask_jsonl_path}, skipping...")
    #             print(f"  [Info] Loaded {len(mask_info)} mask records for {seq_name}.")
    #         else:
    #             mask_info = None  # 若未找到，设置为空
    #             print(f"  [Warning] No mask_info found for {seq_name}, proceeding without mask.")

    #     if run_soi:
    #         print(f"Using `track_and_filter_candidates` for sequence: {seq_name}")
    #         save_path = os.path.join(self.result_dir, seq_name)
    #         # ======== #
    #         # soi run  track and filter
    #         # ======== #
    #         _, _, boxes = tracker.track_and_filter_candidates(
    #             img_files, save_path, anno, seq_name, threshold=threshold,
    #             track_vis=False, heatmap_vis=False, masked=True, save_masked_img=False
    #             )
    #         assert len(boxes) == len(anno)
    #         # 保存筛选后的结果
    #         self._record_wo_times(record_file, boxes) # 不保存时间了
    #     else:
    #         # ======== #
    #         # normal run & mask run
    #         # ======== #
    #         print(f"Using `track` for sequence: {seq_name}")
    #         # track
    #         boxes, times = tracker.track(
    #             seq_name, img_files, anno, visualize=visualize, mask_info=mask_info
    #         )
    #         assert len(boxes) == len(anno)
    #         # 保存常规跟踪结果
    #         self._record(record_file, boxes, times)
            
    #     print(f"  Sequence {seq_name} completed.")

    def run_single_sequence(self, seq_index, tracker, logger, **kwargs):
        """
        在单独的进程中运行单个序列的跟踪任务。

        Args:
            seq_index: 数据集序列的索引。
            tracker: 跟踪器实例。
            **kwargs: 可选参数集合，例如：
                      - logger: 日志
                      - save_dir: 路径
                      - visualize: 是否进行可视化
                      - run_mask: 是否使用遮罩模式
                      - mask_info: 遮罩信息
                      - use_filter: 是否使用 `track_and_filter_candidates`
                      - threshold: track_and_filter_candidates 的阈值
        """
        seq_name = self.dataset.seq_names[seq_index]
        img_files, anno = self.dataset[seq_index]

        print(f"Processing Sequence {seq_index + 1}/{len(self.dataset)}: {seq_name}...")
        logger.info(f"Processing Sequence {seq_index + 1}/{len(self.dataset)}: {seq_name}...")

        # 提取关键字参数
        # save_dir = kwargs.get("save_dir", False)
        visualize = kwargs.get("visualize", False)
        run_mask = kwargs.get("run_mask", False)
        run_soi = kwargs.get("run_soi", False)
        online = kwargs.get("run_mask_online", False)
        threshold = kwargs.get("threshold", 0.25)  # 默认阈值为0.25
        # 构建结果保存路径
        result_dir = self.result_dir_masked if run_mask else self.result_dir
        record_file = os.path.join(result_dir, f"{seq_name}.txt")
        
        mask_info = None
        if run_mask:
            if online:
                pass  # TODO: 处理在线 mask 逻辑
            else:
                # **计算 mask 数据存储目录**
                mask_jsonl_path = os.path.join(self.mask_info_dir, f"{seq_name}_masked_info.jsonl")
                # **确保 mask 信息文件存在**
                assert os.path.exists(mask_jsonl_path), f"Error: Mask info file {mask_jsonl_path} not found!"
                if os.path.exists(mask_jsonl_path):  # 如果 JSONL 文件存在
                    mask_info = []  # 初始化 mask 信息存储列表
                    with open(mask_jsonl_path, "r") as f:
                        for line in f:  # 逐行读取 JSONL 文件
                            try:
                                mask_info.append(json.loads(line.strip()))  # 解析 JSON 并添加到列表
                            except json.JSONDecodeError:
                                logger.error(f" Invalid JSON line in {mask_jsonl_path}, skipping...")
                    logger.info(f" Loaded {len(mask_info)} mask records for {seq_name}.")
                else:
                    mask_info = None  # 若未找到，设置为空
                    logger.info(f" No mask_info found for {seq_name}, proceeding without mask.")

                # **检查 TXT 文件是否存在**
                record_exists = os.path.exists(record_file)
                # **如果 TXT 文件存在，跳过处理**
                if record_exists and run_mask:
                    logger.info(f"  Found valid TXT results for run_mask, skipping {seq_name}")
                    return
        else:
            # **检查 TXT 文件是否存在**
            record_exists = os.path.exists(record_file)
            # **检查 JSONL 文件是否存在，并且行数是否匹配 anno**
            jsonl_file = os.path.join(self.mask_info_dir, f"{seq_name}_mask_info.jsonl")  # JSONL 文件
            jsonl_exists = os.path.exists(jsonl_file)
            jsonl_valid = False

            if jsonl_exists:
                with open(jsonl_file, "r") as f:
                    jsonl_lines = sum(1 for _ in f)  # 计算 JSONL 文件行数
                
                if jsonl_lines == len(anno) - 1:  # JSONL 行数必须匹配 `anno` 长度
                    jsonl_valid = True
                else:
                    logger.warning(f"⚠️  JSONL 文件 {jsonl_file} 记录不完整 (当前: {jsonl_lines} 行, 预期: {len(anno)-1})，删除并重新记录...")
                    os.remove(jsonl_file)  # **删除不完整的 JSONL**
                    jsonl_valid = False  # **强制重新运行**


            # **如果 TXT 文件存在，并且 JSONL 也有效，跳过处理**
            if record_exists and jsonl_valid and run_soi:
                logger.info(f"  Found valid TXT & JSONL results, skipping {seq_name}")
                return
            elif record_exists and run_soi is False:  # 正常运行的判定
                logger.info(f"  Found valid TXT results, skipping {seq_name}")
                return

        # 初始化 tracker
        tracker_cls = choose_tracker(tracker)
        tracker = tracker_cls()

        if run_soi:
            logger.info(f"Using `track_and_filter_candidates` for sequence: {seq_name}")
            print(f"Using `track_and_filter_candidates` for sequence: {seq_name}")
            save_path = os.path.join(self.result_dir, seq_name)
            # ======== #
            # soi run  track and filter
            # ======== #
            _, _, boxes = tracker.track_and_filter_candidates(
                img_files, save_path, anno, seq_name, logger, threshold=threshold,
                track_vis=False, heatmap_vis=False, masked=True, 
                save_masked_img=False, mask_info_dir=self.mask_info_dir,
                )
            assert len(boxes) == len(anno)
            logger.info(f"保存 {seq_name} 筛选后的结果 ")
            # 保存筛选后的结果
            self._record_wo_times(record_file, boxes, logger) # 不保存时间了
            logger.info(f"保存 {seq_name} 筛选后的结果 done ")
        else:
            # ======== #
            # normal run & mask run
            # ======== #
            logger.info(f"Using `track` for sequence: {seq_name}")
            print(f"Using `track` for sequence: {seq_name}")
            # track
            boxes, times = tracker.track(
                seq_name, img_files, anno, logger, visualize=visualize, mask_info=mask_info, 
                mask_vis_dir=os.path.join(result_dir, f"{seq_name}_masked_imgs")  # mask可视化的保存目录
            )
            assert len(boxes) == len(anno)
            # 保存常规跟踪结果
            self._record(record_file, boxes, times)
            
        logger.info(f"  Sequence {seq_name} completed.")
        print(f"  Sequence {seq_name} completed.")


    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt="%.3f", delimiter=",")
        while not os.path.exists(record_file):
            print("warning: recording failed, retrying...")
            np.savetxt(record_file, boxes, fmt="%.3f", delimiter=",")
        print("  Results recorded at", record_file)

        # record running times
        time_dir = os.path.join(record_dir, "times")
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(
            time_dir, os.path.basename(record_file).replace(".txt", "_time.txt")
        )
        np.savetxt(time_file, times, fmt="%.8f")

    def _record_wo_times(self, record_file, boxes, logger, max_retries=5, retry_delay=1):
        """
        记录跟踪结果到文件，确保数据正确保存。

        参数:
            record_file (str): 结果保存路径
            boxes (numpy.ndarray): 需要保存的跟踪结果
            logger (logging.Logger): 记录日志的 logger
            max_retries (int): 最大重试次数，默认 5
            retry_delay (int): 失败后重试的等待时间（秒），默认 1
        """
        # 确保目录存在
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir, exist_ok=True)

        np.savetxt(record_file, boxes, fmt="%.3f", delimiter=",")
        if os.path.exists(record_file):
            logger.info(f"✅ 结果已成功保存: {record_file}")
            return


    def run_soi(
        self, tracker, threshold=0.25, track_vis=False, heatmap_vis=False, masked=True
    ):
        print(
            "Running tracker %s on %s..." % (tracker.name, type(self.dataset).__name__)
        )

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):

            save_path = os.path.join(self.result_dir, self.dataset.seq_names[s])

            seq_name = self.dataset.seq_names[s]
            print("--Sequence %d/%d: %s" % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            num_file = os.path.join(self.score_dir, "nc_%s.txt" % seq_name)
            if os.path.exists(num_file):
                print("  Found results, skipping", seq_name)
                continue

            # tracking loop
            seq_candidate_data, num, boxes = tracker.track_and_filter_candidates(
                img_files,
                save_path,
                anno,
                seq_name,
                threshold=threshold,
                track_vis=False,
                heatmap_vis=False,
                masked=True,
                save_masked_img=False,
            )  # 对齐run_sequence

    def _record_with_score(self, num_file, seq_name, seq_candidate_data, num):
        # record running times
        np.savetxt(num_file, num, delimiter=",", encoding="utf_8_sig", fmt="%d")

        # record score_info
        score_file = os.path.join(self.score_dir, "score_info.jsonl")
        append_to_jsonl_file(score_file, seq_name, seq_candidate_data)

    def sequences_select(
        self, tracker_names, root_dir, save_dir, screen_mode="MaxPooling", th=0.1
    ):
        assert isinstance(tracker_names, (list, tuple))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        name = self.dataset.name
        subset = self.dataset.subset[0]
        info_file = os.path.join(
            save_dir,
            "data",
            self.dataset.name,
            self.subset,
            screen_mode + "th_{}".format(th),
            "%s_%s_info.json" % (name, subset),
        )
        seq_names = []
        for s, (_, anno) in enumerate(self.dataset):  # 序列遍历
            seq_name = self.dataset.seq_names[s]
            seq_tc_num = {}
            soi_frames = []
            # seq_result_select.update({subset: {}})
            for i, tracker in enumerate(tracker_names):  # tracker遍历

                num_csore_json_file = os.path.join(
                    root_dir,
                    "../score_map_json",
                    tracker,
                    self.dataset.name,
                    subset,
                    "score_map",
                    f"{seq_name}_score_map.json",
                )
                if os.path.exists(
                    num_csore_json_file
                ):  # /home/micros/SOI/tomp/lasot/test/score_map/airplane-9_score_map.json
                    print("operate %s in lasot of %s" % (seq_name, tracker))
                else:
                    print("No results for ", seq_name)
                    continue

                # 打开json文件，读取数据
                with open(num_csore_json_file, "r") as f:
                    data = json.load(f)
                _, _, seq_tc_num[i] = extract_candidate_set(data, anno)

            for j in range(len(anno) - 1):
                if np.any(np.isnan(anno[j + 1])):
                    print("%s is absent frame")
                    continue
                a = 0
                for k in range(len(seq_tc_num)):
                    if seq_tc_num[k][j] > 1:
                        a += 1
                if a > 1:
                    soi_frames.append(j + 1)

            if len(soi_frames) > 0:  # 存在soi
                self.choose_seq_for_soi(seq_name, soi_frames, save_dir, screen_mode, th)
                # append
                seq_names.append(seq_name)
            else:
                continue

        with open(info_file, "w") as f:
            json.dump(seq_names, f, indent=4)

    def sequences_analysis(
        self, tracker_names, root_dir, save_dir, screen_mode="MaxPooling", th=0.1
    ):
        assert isinstance(tracker_names, (list, tuple))
        print("th is:" + str(th))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        name = self.dataset.name
        subset = self.dataset.subset[0]
        info_file = os.path.join(
            save_dir,
            "data",
            self.dataset.name,
            self.subset,
            screen_mode + "th_{}".format(th),
            "%s_%s_info.json" % (name, subset),
        )
        seq_names = []
        for s, (_, anno) in enumerate(self.dataset):  # 序列遍历
            seq_name = self.dataset.seq_names[s]
            seq_trackers_judge = 0
            ratio = {}
            # seq_result_select.update({subset: {}})
            for i, tracker in enumerate(tracker_names):  # tracker遍历

                num_csore_json_file = os.path.join(
                    root_dir,
                    tracker,
                    self.dataset.name,
                    subset,
                    "score_map",
                    f"{seq_name}_score_map.json",
                )
                if os.path.exists(
                    num_csore_json_file
                ):  # /home/micros/SOI/tomp/lasot/test/score_map/airplane-9_score_map.json
                    print("operate %s in lasot of %s" % (seq_name, tracker))
                else:
                    print("No results for ", seq_name)
                    continue

                # 打开json文件，读取数据
                with open(num_csore_json_file, "r") as f:
                    data = json.load(f)
                seq_candidate_scores, seq_candidate_coords, seq_tc_num = (
                    analysis_candidate_set(
                        data, anno, th=th, screen_mode=screen_mode, alpha=0.8
                    )
                )

                single_boj_num = 0
                obj_disappear_num = 0
                has_candiadates = 0
                for j, num in enumerate(seq_tc_num):  # 帧遍历
                    if num == 1:
                        single_boj_num += 1
                    elif num == 0:
                        obj_disappear_num += 1
                        # print('object disappearing for seq %s' % seq_name)
                    else:
                        has_candiadates += 1
                if has_candiadates == 0:  # 无干扰物
                    ratio[tracker] = 0.0
                else:
                    seq_trackers_judge += 1
                    ratio[tracker] = has_candiadates
            if seq_trackers_judge > 1:
                ratio = self.mk_ratio(ratio, len(anno))
                if ratio["adv"] >= 0.1:
                    self.choose_seq_for_soi(
                        seq_name, ratio, save_dir, th=th, screen_mode=screen_mode
                    )
                    # append
                    seq_names.append(seq_name)
                else:
                    print(
                        "Interferer challenge for sequence %s is not dominant"
                        % seq_name
                    )
            else:
                print("The sequence %s has not interference" % seq_name)
        #
        with open(info_file, "w") as f:
            json.dump(seq_names, f, indent=4)

    def choose_seq_for_soi(self, seq_name, soi_frames, save_dir, screen_mode, th):
        # seq_dir = os.path.join(save_dir, 'data')
        # anno_dir = os.path.join(save_dir, 'data')
        base_path = os.path.dirname(self.dataset.seq_dirs[0])
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(base_path)),
            seq_name[: seq_name.rfind("-")],
            seq_name,
        )
        save_path = os.path.join(
            save_dir,
            "data",
            self.dataset.name,
            self.subset,
            screen_mode + "th_{}".format(th),
            "data",
            seq_name,
        )
        print(
            f"choose seq {file_path} into soi"
        )  # 使用f-string来格式化字符串，更简洁高效

        if os.path.exists(save_path):
            print("seq has been operated")
        else:
            makedir(save_path)
            frame_path = os.path.join(save_path, "soi.txt")
            np.savetxt(frame_path, soi_frames, fmt="%d", delimiter=",")

    def _record_with_score_map(self, seq_name, seq_score_map_data):
        #
        json_file = os.path.join(self.score_map_dir, f"{seq_name}_score_map.json")
        # 将响应图添加到字典中
        data = {"score_map_list": seq_score_map_data["sm"]}
        # 打开json文件，写入数据
        with open(json_file, "w") as f:
            json.dump(data, f)
