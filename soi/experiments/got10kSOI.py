from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2

from ..datasets import GOT10kSOI
from ..utils.metrics import rect_iou
from ..utils.viz import show_frame
from ..utils.ioutils import compress
from ..utils.screening_util import *
from ..utils.help import makedir


def call_back():
    return ExperimentGOT10kSOI


class ExperimentGOT10kSOI(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run experiments on
            sequences specified by this file.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, save_dir, subset='val', list_file=None, th=0.1,
                 use_dataset=True):  # list_file 要修改到got10kSOI路径上
        super(ExperimentGOT10kSOI, self).__init__()
        assert subset in ['train', 'val', 'test']
        self.subset = subset
        if use_dataset:
            self.dataset = GOT10kSOI(root_dir, subset=subset, list_file=list_file, th=th)
        self.nbins_iou = 101
        self.repetitions = 3

        '''
        -save_dir
            |--TRACKER_NAME
                |---DATASET
                    |--result
                    |--reports
                    |--image
                    |--scores
        '''
        self.result_dir = os.path.join(save_dir, 'results')
        self.report_dir = os.path.join(save_dir, 'reports')
        self.score_dir = os.path.join(save_dir, 'scores')
        self.img_dir = os.path.join(save_dir, 'image')
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.score_dir)
        makedir(self.img_dir)

    def run(self, tracker, visualize=False, save_video=False, overwrite_result=True):
        if self.subset == 'test':
            print('\033[93m[WARNING]:\n' \
                  'The groundtruths of GOT-10k\'s test set is withholded.\n' \
                  'You will have to submit your results to\n' \
                  '[http://got-10k.aitestunion.com/]' \
                  '\nto access the performance.\033[0m')
            time.sleep(2)

        print('Running tracker %s on GOT-10k...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                        tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file) and not overwrite_result:
                    print('  Found results, skipping', seq_name)
                    continue

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=visualize)

                # record results
                self._record(record_file, boxes, times)

            # save videos
            if save_video:
                video_dir = os.path.join(os.path.dirname(os.path.dirname(self.result_dir)),
                                         'videos', 'GOT-10k', tracker.name)
                video_file = os.path.join(video_dir, '%s.avi' % seq_name)

                if not os.path.isdir(video_dir):
                    os.makedirs(video_dir)
                image = Image.open(img_files[0])
                img_W, img_H = image.size
                out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 10, (img_W, img_H))
                for ith, (img_file, pred) in enumerate(zip(img_files, boxes)):
                    image = Image.open(img_file)
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')
                    img = np.array(image)[:, :, ::-1].copy()
                    pred = pred.astype(int)
                    cv2.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), self.color['pred'],
                                  2)
                    if ith < anno.shape[0]:
                        gt = anno[ith].astype(int)
                        cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), self.color['gt'], 2)
                    out_video.write(img)
                out_video.release()
                print('  Videos saved at', video_file)

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple, str))

        if self.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = os.path.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                compress('.', save_file)
                print('Records saved at', save_file + '.zip')

            # print submission guides
            print('\033[93mLogin and follow instructions on')
            print('http://got-10k.aitestunion.com/submit_instructions')
            print('to upload and evaluate your tracking results\033[0m')

            # switch back to previous working directory
            os.chdir(pwd)

            return None
        elif self.subset == 'val':

            if isinstance(tracker_names, str):
                print('tracker_names is path to results')
                tracker_names = os.listdir(tracker_names)

            # meta information is useful when evaluation
            self.dataset.return_meta = True

            # assume tracker_names[0] is your tracker
            report_dir = os.path.join(self.report_dir)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names
            covers = {s: self.dataset[s][2]['cover'][1:] for s in seq_names}

            performance = {}
            for name in tracker_names:
                if name == 'KeepTrack*' or name == 'UAV-KT*' \
                        or name == 'tompKT' or name == 'UAV-KT' or name == 'JointNLT' or name == 'PrDiMP':
                    continue  # name == 'transKT' or
                if not os.path.exists(os.path.join('/mnt/second/wangyipei/SOI/tracker_result', name, 'got10kSOI')):
                    continue
                print('Evaluating', name)
                ious = {}
                times = {}
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

                for s, (_, anno, meta) in enumerate(self.dataset):
                    seq_name = self.dataset.seq_names[s]
                    # record_files = glob.glob(os.path.join(
                    #     self.result_dir, name, seq_name,
                    #     '%s_[0-9]*.txt' % seq_name))
                    base_path = os.path.join(self.result_dir[:self.result_dir.find('SOI') + 3], 'tracker_result', name,
                                             self.dataset.name,
                                             self.result_dir[self.result_dir.find(self.dataset.subset[0]):], name)
                    if name == 'OSTrack' or name == 'ToMP' or name == 'transKT':
                        record_files = glob.glob(os.path.join(base_path, '%s.txt' % seq_name))
                    else:
                        record_files = glob.glob(os.path.join(base_path, '%s_[0-9]*.txt' % seq_name))
                    if len(record_files) == 0:
                        raise Exception('Results for sequence %s not found.' % seq_name)

                    # read results of all repetitions
                    if name == 'OSTrack' or name == 'ToMP' or name == 'transKT':
                        boxes = [np.loadtxt(f, delimiter='\t') for f in record_files]
                    else:
                        boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                    assert all([b.shape == anno.shape for b in boxes])

                    # calculate and stack all ious
                    bound = ast.literal_eval(meta['resolution'])
                    seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]
                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ious = np.concatenate(seq_ious)
                    ious[seq_name] = seq_ious

                    # stack all tracking times
                    times[seq_name] = []
                    # time_file = os.path.join(
                    #     self.result_dir, name, seq_name,
                    #     '%s_time.txt' % seq_name)
                    time_file = os.path.join(base_path, '{}_time.txt'.format(seq_name))
                    if os.path.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times
                    else:
                        seq_times = []

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                ious = np.concatenate(list(ious.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, succ_curve = self._evaluate(ious, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            # plot success curves
            if plot_curves:
                self.plot_curves([report_file], tracker_names)

            return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot' + extension)
        key = 'overall'

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def run_for_Data_Screening(self, tracker):
        print('Running tracker %s on %s...' % (tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):

            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            num_file = os.path.join(self.score_dir, 'nc_%s.txt' % seq_name)
            if os.path.exists(num_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            seq_candidate_data, num = tracker.track_for_screening(img_files, anno)  # 对齐run_sequence
            # assert len(seq_candidate_data['index']) + 1 == len(anno)

            # record results
            self._record_with_score(num_file, seq_name, seq_candidate_data, num)

    def _record_with_score(self, num_file, seq_name, seq_candidate_data, num):
        # record soi num
        np.savetxt(num_file, num, delimiter=',', encoding='utf_8_sig', fmt='%d')

        # record score_info
        score_file = os.path.join(self.score_dir, 'score_info.json')
        dump_seq_data_to_disk(score_file, seq_name, seq_candidate_data)

    def sequences_select(self, tracker_names, root_dir, save_dir):
        assert isinstance(tracker_names, (list, tuple))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        name = self.dataset.name
        subset = self.dataset.subset
        seq_names = []
        for s, (_, anno) in enumerate(self.dataset):  # 序列遍历
            seq_name = self.dataset.seq_names[s]
            seq_trackers_judge = 0
            ratio = {}
            # seq_result_select.update({subset: {}})
            for i, tracker in enumerate(tracker_names):  # tracker遍历

                num_file = os.path.join(root_dir, tracker, name, subset, 'scores', 'nc_%s.txt' % seq_name)
                if os.path.exists(num_file):
                    print('operate %s in lasot of %s' % (seq_name, tracker))
                else:
                    print('No results for ', seq_name)
                    continue

                score_num = np.loadtxt(num_file, dtype=float, delimiter=',', encoding='utf_8_sig')

                single_boj_num = 0
                obj_disappear_num = 0
                has_candiadates = 0
                for j, num in enumerate(score_num):  # 帧遍历
                    if num == 1:
                        single_boj_num += 1
                    elif num == 0:
                        obj_disappear_num += 1
                        # print('object disappearing for seq %s' % seq_name)
                    else:
                        has_candiadates += 1
                if has_candiadates == 0:  # 无干扰物
                    ratio[tracker] = 0.
                else:
                    seq_trackers_judge += 1
                    ratio[tracker] = has_candiadates
            if seq_trackers_judge > 1:
                ratio = self.mk_ratio(ratio, len(anno))
                if ratio['adv'] >= 0.1:
                    self.choose_seq_for_soi(seq_name, ratio, save_dir)
                    # append
                    seq_names.append(seq_name)
                else:
                    print('Interferer challenge for sequence %s is not dominant' % seq_name)
            else:
                print('The sequence %s has not interferer challenge' % seq_name)

        info_file = os.path.join(save_dir, self.dataset.name, 'data', self.subset, 'list.txt')
        if not os.path.isdir(info_file):
            os.makedirs(info_file)
        np.savetxt(info_file, seq_names, fmt='%s', delimiter='\n')

    def choose_seq_for_soi(self, seq_name, ratio, save_dir):
        # seq_dir = os.path.join(save_dir, 'data')
        # anno_dir = os.path.join(save_dir, 'data')

        base_path = os.path.dirname(self.dataset.seq_dirs[0])
        file_path = os.path.join(base_path, seq_name)
        print(f"choose seq {file_path} into soi")  # 使用f-string来格式化字符串，更简洁高效
        save_path = os.path.join(save_dir, self.dataset.name, 'data', self.subset, seq_name)
        if os.path.exists(save_path):
            print('seq has been operated')
        else:
            shutil.copytree(file_path, save_path)

            ra_path = os.path.join(save_path, 'rate_interference.txt')
            for item in ratio.items():
                for i in range(len(item)):
                    str1 = item[i]
                    with open(ra_path, 'a') as f:
                        f.write(str(str1) + '\n')  # 使用\n代替\r来换行，更通用

    def mk_ratio(self, ratio, length):
        # use list comprehension to calculate the ratio for each key
        ratio1 = {k + '_ra': v / length for k, v in ratio.items()}
        # use sum and len to calculate the average ratio
        ra = sum(ratio.values()) / (len(ratio) * length)
        print(ra)
        # use dict.update to add new items to the original ratio
        ratio.update({'adv': ra, 'length': length})
        ratio.update(ratio1)
        return ratio
