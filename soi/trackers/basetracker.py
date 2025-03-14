from __future__ import absolute_import
from typing import Union
import torch
import numpy as np
import time
import cv2 as cv
from collections import defaultdict
from soi.utils.screening_util import *
from ..utils.help import makedir
from ..utils.metrics import iou


class Tracker(object):
    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
        if self.is_using_cuda:
            # print("Detect the CUDA devide")
            self._timer_start = torch.cuda.Event(enable_timing=True)
            self._timer_stop = torch.cuda.Event(enable_timing=True)
        self._timestamp = None

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    @property
    def is_using_cuda(self):
        self.cuda_num = torch.cuda.device_count()
        if self.cuda_num == 0:
            return False
        else:
            return True

    def _start_timing(self) -> Union[float, None]:
        if self.is_using_cuda:
            self._timer_start.record()
            timestamp = None
        else:
            timestamp = time.time()
            self._timestamp = timestamp
        return timestamp

    def _stop_timing(self) -> float:
        if self.is_using_cuda:
            self._timer_stop.record()
            torch.cuda.synchronize()
            # cuda event record return duration in milliseconds.
            duration = self._timer_start.elapsed_time(self._timer_stop)
            duration /= 1000.0
        else:
            duration = time.time() - self._timestamp
        return duration

    def track(self, seq_name, img_files, anno, logger, visualize=False, mask_vis_dir=None, save_img=False, mask_info=None):
        """
        目标跟踪主函数，遍历视频帧并更新目标位置，可选可视化或保存结果。

        参数：
            seq_name (str): 序列名称。
            img_files (list): 视频帧文件路径列表。
            anno (numpy.ndarray): 目标在每帧的真实标注框 (GT) [x, y, w, h]。
            visualize (bool): 是否实时显示跟踪结果，默认 False。
            mask_vis_dir (str): mask 可视化的保存目录，若为 None，则不保存。
            save_img (bool): 是否保存跟踪可视化结果，默认 False。
            mask_info (dict): 预先计算的遮挡信息，用于对特定目标区域进行遮挡。

        返回：
            boxes (numpy.ndarray): 记录跟踪的目标框信息，shape=(frame_num, 4)。
            times (numpy.ndarray): 记录每帧跟踪时间，shape=(frame_num,)。
        """
        frame_num = len(img_files)
        box = anno[0, :]  # 第一帧 GT 目标框
        boxes = np.zeros((frame_num, 4))  # 记录跟踪结果
        boxes[0] = box
        times = np.zeros(frame_num)  # 存储每帧跟踪时间

        for f, img_file in enumerate(img_files):
            image = cv.imread(img_file)
            height, width = image.shape[:2]
            img_resolution = (width, height)

            if f == 0:
                self._start_timing()
                self.init(image, box)
                times[f] = self._stop_timing()
            else:
                if mask_info:
                    mask_boxes, gt_coord = find_mask_info_for_frame(mask_info, seq_name, f)
                    if isinstance(mask_boxes, list) and mask_boxes:
                        image = mask_image_with_boxes(image, mask_boxes, gt_coord, 
                                                    debug_save_path=f"{mask_vis_dir}/{f:06d}.jpg")

                # 执行目标跟踪
                self._start_timing()
                frame_box = self.update(image)
                frame_box = np.rint(frame_box)
                boxes[f, :] = frame_box
                times[f] = self._stop_timing()

                # **设置日志间隔**
                log_interval = 100
                if f % log_interval == 0 or f == frame_num - 1:
                    logger.info(f"{seq_name} {self.name} Tracking {f}/{frame_num - 1} {str(frame_box)}")

                # **可视化或保存结果**
                if save_img and f % 2 == 0:
                    frame_disp = visualize_tracking(seq_name, f, image, frame_box, anno[f, :], img_resolution)
                    save_path = f"{mask_vis_dir}/{f:06d}.jpg"
                    cv.imwrite(save_path, frame_disp)

        return boxes, times

    def track_and_filter_candidates(self, img_files, save_path, anno, seq_name, logger, 
                                    mask_info_dir=None, threshold=0.1,
                                    track_vis=False, heatmap_vis=False, masked=True, save_masked_img=False):
        """
        根据目标框在视频序列中进行追踪，并筛选出包含目标候选物的帧数据。

        参数：
            img_files (list): 图像文件路径列表。
            anno (numpy.ndarray): 第一个帧的目标框信息。
            threshold (float): 筛选候选物体的得分阈值，默认值为0.1。

        返回：
            seq_candidate_data (defaultdict): 包含每帧候选数据的字典。
            scores (numpy.ndarray): 每帧的候选数目（得分）。
        """
        frame_num = len(img_files)
        box = anno[0, :]  # 获取第一帧的目标框信息
        boxes = np.zeros((frame_num, 4))  # save the tracking result
        boxes[0] = box
        candidate_num = np.zeros((frame_num, 1), dtype=int)  # 保存每帧的追踪结果（候选物体数量）
        candidate_num[0] = 1  # 第一帧的初始候选物个数为1
        seq_candidate_data = defaultdict(list)  # 用于存储所有帧的候选数据


        for f, img_file in enumerate(img_files):
            image = cv.imread(img_file)
            height, width = image.shape[:2]
            img_resolution = (width, height)

            if f == 0:
                self.init(image, box)  # 初始化追踪器，第一帧
            else:
                frame_box = self.update(image)  # 更新追踪器，进行追踪

                # 获取当前帧的候选数据和候选物体数量
                frame_candidate_data = extract_candidate_data(self.tracker.distractor_dataset_data, th=threshold)
                # 更新序列数据，加入当前帧的候选数据
                update_seq_data(seq_candidate_data, frame_candidate_data)

                frame_box = np.rint(frame_box)
                frame_box = np.array(frame_box)
                track_result = frame_box.reshape((1, 4))
                boxes[f, :] = frame_box 

                current_gt = anno[f, :].reshape((1, 4))
                bound = img_resolution
                seq_iou = iou(current_gt, track_result, bound=bound)
                # print(seq_name, self.name, " Tracking %d/%d" % (f, frame_num - 1), frame_box)
                log_interval = 100  # 设定日志间隔（每 50 帧打印一次）
                if f % log_interval == 0 or f == frame_num - 1:  # 每 `log_interval` 帧打印一次，确保最后一帧一定打印
                    logger.info(f"{seq_name} {self.name} Tracking {f}/{frame_num - 1} {str(frame_box)}")
                # 更新当前帧的候选物体得分
                candidate_num[f] = frame_candidate_data["tg_num"]

                if track_vis:
                    visualize_tracking(image, anno[f, :], frame_box, frame_candidate_data, seq_iou, save_path, f, img_resolution)

                if heatmap_vis:
                    visualize_heatmap(frame_candidate_data, save_path, f)

                if masked:
                    process_masked_images(seq_name, image, anno[f, :], frame_candidate_data, track_result, mask_info_dir, f, save_masked_img)

            # 可选：启用按键中断（例如按'q'退出）
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break

        return seq_candidate_data, candidate_num, boxes


def visualize_tracking_for_track(seq_name, frame_id, image, frame_box, gt_box, img_resolution):
    """
    处理目标跟踪的可视化，包括绘制预测框、GT 目标框，并在图像上叠加相关信息。

    参数：
        seq_name (str): 序列名称。
        frame_id (int): 当前帧索引。
        image (numpy.ndarray): 当前帧图像。
        frame_box (numpy.ndarray): 预测框 [x, y, w, h]。
        gt_box (numpy.ndarray): 真实 GT 目标框 [x, y, w, h]。
        img_resolution (tuple): 图像的 (width, height)。

    返回：
        frame_disp (numpy.ndarray): 处理后的可视化图像。
    """
    frame_disp = image.copy()
    height, width = img_resolution
    state = [int(s) for s in frame_box]

    # 确保目标框坐标在图像范围内
    state[0] = max(0, state[0])
    state[1] = max(0, state[1])
    state[2] = min(width - state[0], state[2])
    state[3] = min(height - state[1], state[3])

    gt = [int(s) for s in gt_box]
    font_face = cv.FONT_HERSHEY_SIMPLEX

    # 显示当前帧编号
    cv.putText(frame_disp, f"No.{frame_id:06d}", (50, 100), font_face, 0.8, (0, 255, 0), 2)

    # 计算 IoU 并显示
    if (gt_box != np.array([0, 0, 0, 0])).all():
        iou_value = iou(gt_box.reshape((1, 4)), frame_box.reshape((1, 4)), bound=img_resolution)
        cv.putText(frame_disp, f"seq iou: {iou_value:.2f}", (50, 130), font_face, 0.8, (0, 255, 0), 2)

    # 绘制预测框（绿色）
    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), (0, 255, 0), 5)

    # 绘制 GT 目标框（红色）
    cv.rectangle(frame_disp, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 0, 255), 5)

    return frame_disp


def visualize_tracking(image, gt_box, frame_box, frame_candidate_data, seq_iou, img_path, frame_id, img_resolution):
    frame_disp = image.copy()
    font_face = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame_disp, "No.%06d" % frame_id, (0, 20), font_face, 0.8, (0, 255, 0), 2)
    
    if (gt_box != np.array([0, 0, 0, 0])).all():  # (anno[f, :] != np.array([0, 0, 0, 0])).all():
        cv.putText(frame_disp, "seq iou: %2f" % seq_iou, (0, 50), font_face, 0.8, (0, 255, 0), 2)
    # predict box
    state = [int(s) for s in frame_box]
    state[0] = 0 if state[0] < 0 else state[0]
    state[1] = 0 if state[1] < 0 else state[1]
    state[2] = (img_resolution[0] - state[0] if state[0] + state[2] > img_resolution[0] else state[2])
    state[3] = (img_resolution[1] - state[1] if state[1] + state[3] > img_resolution[1] else state[3])
    cv.rectangle(frame_disp, (state[0], state[1]), 
                 (state[2] + state[0], state[3] + state[1]), (0, 0, 255), 2)  # (0, 0, 255)是红色
    # gt
    gt = [int(s) for s in gt_box]
    cv.rectangle(frame_disp, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 255, 0), 2)  # 0,255,0是绿色
    # candidate boxes
    for i, box in enumerate(frame_candidate_data["candidate_boxes"]):
        # if i == 0:
        #     continue
        temp_state = [int(s) for s in box.squeeze(0)]
        temp_state[0] = 0 if temp_state[0] < 0 else temp_state[0]
        temp_state[1] = 0 if temp_state[1] < 0 else temp_state[1]
        temp_state[2] = (img_resolution[0] - temp_state[0] if temp_state[0] + temp_state[2] > img_resolution[0] else temp_state[2])
        temp_state[3] = (img_resolution[1] - temp_state[1] if temp_state[1] + temp_state[3] > img_resolution[1] else temp_state[3])
        temp_iou = iou(frame_box.reshape((1, 4)), box.cpu().numpy(), bound=img_resolution)
        if temp_iou > 0.6:
            continue
        # 画出候选框 虚线灰色
        cv.rectangle(frame_disp, (temp_state[0], temp_state[1]),
                      (temp_state[2] + temp_state[0], temp_state[3] + temp_state[1]), (200, 200, 200), 2)  # 黄色(0, 255, 255)
        # 搜索区域
        search_area_box = frame_candidate_data["search_area_box"]
        cv.rectangle(
            frame_disp, 
            (int(search_area_box[0]), int(search_area_box[1])), 
            (int(search_area_box[2]) + int(search_area_box[0]), int(search_area_box[3]) + int(search_area_box[1])), 
            (255, 0, 0), 
            2
        )
    save_path = f"{img_path}/img/{frame_id:06d}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path, frame_disp)


def visualize_heatmap(frame_candidate_data, img_path, frame_id):
    frame_search_img = (frame_candidate_data["search_img"].squeeze().cpu().numpy())
    # 归一化热力图并转换为颜色映射
    heatmap = frame_candidate_data["score_map"].squeeze().cpu().numpy()  # self.tracker.distractor_dataset_data["score_map"]
    heatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    # 如果尺寸不一致，调整尺寸
    if (heatmap.shape[:2] != frame_search_img.shape[:2]):  
        heatmap = cv.resize(heatmap, (frame_search_img.shape[1], frame_search_img.shape[0]))
    # 融合热力图与原图
    heat_map_search_img = cv.addWeighted(heatmap, 0.7, frame_search_img, 0.3, 0)
    # 添加文本
    fontScale = (frame_search_img.shape[1] / 500 * 1.0)  # 通过图像高度动态设置字体大小
    target_candidate_scores = frame_candidate_data["target_candidate_scores"]
    font_face = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(heat_map_search_img, "No.%06d" % (frame_id), (10, 20), font_face, 0.8, (0, 255, 0), 2)
    # 将 target_candidate_scores 转换为可显示的字符串
    if target_candidate_scores.numel() == 1:  # 只有一个元素
        score_text = "tc_scores: %.2f" % target_candidate_scores.item()
    else:  # 多个元素，转换为逗号分隔的字符串
        score_list = (target_candidate_scores.flatten().tolist())  # 转换为 Python 列表
        score_text = "tc_scores: " + ", ".join(["%.2f" % s for s in score_list])
    # 在图像上显示分数文本
    cv.putText(heat_map_search_img, score_text, (10, 50), font_face, 0.8, (0, 255, 0), 2)
    # 保存图像
    ca_save_path = f"{img_path}/search_img/{frame_id:06d}.jpg"
    os.makedirs(os.path.dirname(ca_save_path), exist_ok=True)
    cv.imwrite(ca_save_path, heatmap)


def process_masked_images(seq_name, image, gt_box, frame_candidate_data, track_result, mask_path, frame_id, save_masked_img):
    """ 处理 Masked 图像，并确保 `masked_info.jsonl` 追加写入不会重复 """
    
    if save_masked_img:
        masked_save_path = f"{mask_path}/masked_img/{seq_name}/{frame_id:08d}.jpg"
        os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)
        mask_image_with_boxes(image, gt_box, frame_candidate_data["candidate_boxes"], track_result, 
                              iou_threshold=0.7, fill_color=(0, 0, 0), need_save=True, save_path=masked_save_path)
    else:
        masked_save_path = os.path.join(mask_path, f"{seq_name}_mask_info.jsonl")
        existing_frames = set()

        # **检查 `masked_info.jsonl` 是否已有该帧数据**
        if os.path.exists(masked_save_path):
            with open(masked_save_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing_frames.add(record.get("frame_id"))
                    except json.JSONDecodeError:
                        continue  # 忽略 JSON 解析错误的行
        
        # **如果 `frame_id` 已存在，跳过写入**
        if frame_id in existing_frames:
            print(f"⚠️ Frame {frame_id} already exists in {masked_save_path}, skipping write.")
            return

        # **否则，追加写入**
        process_candidate_boxes_and_gt(
            seq_name, masked_save_path, frame_id, gt_box, 
            frame_candidate_data["candidate_boxes"], 
            iou_threshold=0.5, bound=(image.shape[1], image.shape[0])
        )


def process_candidate_boxes_and_gt(seq_name, masked_save_path, frame_id, gt, candidate_boxes, iou_threshold=0.5, bound=None):
    """ 处理候选框和 GT 框，确定哪些需要 Mask，按 JSONL 格式存储，每行代表一帧。 """
    track_box = [int(s) for s in candidate_boxes[0].squeeze(0)]  # 获取预测框（第一个候选框）
    other_box_tensors = candidate_boxes[1:]  # 其余候选框
    gt = [int(s) for s in gt]  # 处理 GT（真实目标框），转换格式
    
    # 计算预测框与 GT 的 IoU（交并比）
    track_iou_gt = iou4list(track_box, gt)
    
    # 计算其他候选框与 GT 的 IoU
    other_iou_gts = [iou4list([int(s) for s in box.squeeze(0)], gt) for box in other_box_tensors]
    
    # 判断是否存在 IoU 高于阈值的正确候选框
    exist_other_good_box = any(val >= iou_threshold for val in other_iou_gts)
    
    # 初始化需要 Mask 的框
    mask_boxes = []

    # ------------------------------ 
    # 根据不同情况决定 Mask 逻辑 
    # ------------------------------
    if track_iou_gt >= iou_threshold:  # 预测框 IoU 高，可能是正确或妥协
        status = "Compromise" if exist_other_good_box else "Correct"
    else:  # 预测框 IoU 低，可能漂移或失败
        mask_boxes.append({"x1": track_box[0], "y1": track_box[1], "x2": track_box[0] + track_box[2], "y2": track_box[1] + track_box[3]})
        if exist_other_good_box:  # 存在正确候选框，跟踪漂移
            status = "Drift"
            for box_tensor in other_box_tensors:  # 遮挡所有候选框
                ob = [int(s) for s in box_tensor.squeeze(0)]
                mask_boxes.append({"x1": ob[0], "y1": ob[1], "x2": ob[0] + ob[2], "y2": ob[1] + ob[3]})
        else:  # 无正确候选框，跟踪失败
            status = "Fail"

    # 记录 GT 框信息 
    gt_coord = {"x1": gt[0], "y1": gt[1], "x2": gt[0] + gt[2], "y2": gt[1] + gt[3]}
    # 组织 JSON 数据
    frame_mask_info = {"frame_id": frame_id, "status": status, "track_iou_gt": track_iou_gt, "gt_coord": gt_coord, "mask_boxes": mask_boxes}
    # 以 JSONL 方式保存到文件 
    os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)  # 确保目录存在
    with open(masked_save_path, "a") as f: f.write(json.dumps(frame_mask_info) + "\n")  # 追加写入 JSONL 文件
