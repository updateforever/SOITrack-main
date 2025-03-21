import cv2 as cv
import numpy as np
import os


def makedir(path):
    """根据指定路径创建文件夹"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def read_filename(path):
    """返回指定路径下的文件名称"""
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filenames.append(os.path.join(root, file))
    return sorted(filenames)


def draw_videocube_demo(seq_name):
    """绘制biodrone的demo"""
    save_dir = '/mnt/second/wangyipei/SOI/demo'
    # 改每个baseline里面的路径以及序列名字就行
    demo_seqs = {
        'videocube':
            {
                'image_path': '/mnt/first/hushiyu/SOT/VideoCube/data/val/%s/frame_%s' % (seq_name, seq_name,),
                'gt_path': '/mnt/first/hushiyu/SOT/VideoCube/data/val/%s/result_%s.txt' % (seq_name, seq_name),
                'start_frame': None,  # 起始帧，默认为空
                'end_frame': None,
                'interval': 1,  # 绘制间隔
                'baseline1': '/mnt/second/wangyipei/SOI/tracker_result/transKT/videocubeSOI/val/results/transKT_restart/%s.txt' % seq_name,
                'baseline2': '/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/videocubeSOI/val/results/KeepTrack_restart/%s.txt' % seq_name,
                'baseline3': '/mnt/second/wangyipei/SOI/tracker_result/ToMP/videocubeSOI/val/results/ToMP_restart/%s.txt' % seq_name,
                'baseline4': '/mnt/second/wangyipei/SOI/tracker_result/OSTrack/videocubeSOI/val/results/OSTrack_restart/%s.txt' % seq_name,
                # baseline4用紫色
            },
    }

    thickness = 5

    for key, val in demo_seqs.items():
        save_seq_dir = os.path.join(save_dir, key, '%s' % seq_name)
        if os.path.exists(save_seq_dir):
            print('has done')
            break
        makedir(save_seq_dir)

        filenames = read_filename(val['image_path'])

        gts = np.loadtxt(val['gt_path'], delimiter=',')

        baseline1s = np.loadtxt(val['baseline1'], delimiter=',')
        baseline2s = np.loadtxt(val['baseline2'], delimiter=',')
        baseline3s = np.loadtxt(val['baseline3'], delimiter=',')
        baseline4s = np.loadtxt(val['baseline4'], delimiter=',')

        # if start_frame != None or end_frame != None:
        #     gts = gts[start_frame:end_frame]
        #     baseline1s = baseline1s[start_frame:end_frame]
        #     baseline2s = baseline2s[start_frame:end_frame]
        #     baseline3s = baseline3s[start_frame:end_frame]
        #     filenames = filenames[start_frame:end_frame]

        print(len(gts), len(filenames), len(baseline1s), len(baseline2s), len(baseline3s), len(baseline4s))

        assert (len(gts) == len(filenames) == len(baseline1s) == len(baseline2s) == len(baseline3s))

        i = 0
        while i < len(filenames):
            filename = filenames[i]

            name = filename.split('/')[-1]
            # print(name)

            gt = [int(num) for num in gts[i]]
            baseline1 = [int(num) for num in baseline1s[i]]
            baseline2 = [int(num) for num in baseline2s[i]]
            baseline3 = [int(num) for num in baseline3s[i]]
            baseline4 = [int(num) for num in baseline4s[i]]

            img = cv.imread(filename)

            cv.rectangle(img, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 255, 0), thickness)  # gt用绿色
            cv.rectangle(img, (baseline1[0], baseline1[1]), (baseline1[2] + baseline1[0], baseline1[3] + baseline1[1]),
                         (0, 0, 255), thickness)  # baseline1用红色
            cv.rectangle(img, (baseline2[0], baseline2[1]), (baseline2[2] + baseline2[0], baseline2[3] + baseline2[1]),
                         (255, 0, 0), thickness)  # baseline2用蓝色
            cv.rectangle(img, (baseline3[0], baseline3[1]), (baseline3[2] + baseline3[0], baseline3[3] + baseline3[1]),
                         (0, 255, 255), thickness)  # baseline3用黄色
            cv.rectangle(img, (baseline4[0], baseline4[1]), (baseline4[2] + baseline4[0], baseline4[3] + baseline4[1]),
                         (255, 0, 255), thickness)  # baseline4用紫色

            cv.imwrite(os.path.join(save_seq_dir, name), img)

            i += int(val['interval'])


def draw_lasot_demo(seq_name):
    """绘制demo"""
    save_dir = '/home/wyp/project/SOITrack-main/output/lasot_demo'
    # 改每个baseline里面的路径以及序列名字就行
    # demo_seqs = {
    #     'lasot':
    #         {
    #             'image_path': '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s' % (seq_name.split('-')[0], seq_name),
    #             'gt_path': '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s/groundtruth.txt' % (
    #                 seq_name.split('-')[0], seq_name),
    #             'start_frame': None,  # 起始帧，默认为空
    #             'end_frame': None,
    #             'interval': 1,  # 绘制间隔
    #             'baseline1': '/mnt/second/wangyipei/SOI/tracker_result/TransKT/lasotSOI/test/results/TransKT/%s.txt' % seq_name,
    #             # baseline1用红色
    #             'baseline2': '/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/lasotSOI/test/results/KeepTrack/%s.txt' % seq_name,
    #             # baseline2用蓝色
    #             'baseline3': '/mnt/second/wangyipei/SOI/tracker_result/ToMP/lasotSOI/test/results/ToMP/%s.txt' % seq_name,
    #             # baseline3用黄色
    #             'baseline4': '/mnt/second/wangyipei/SOI/tracker_result/OSTrack/lasotSOI/test/results/OSTrack/%s.txt' % seq_name,
    #             # baseline4用紫色
    #         },
    # }
    demo_seqs = {
        'lasot':
            {
                'image_path': '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s' % (seq_name.split('-')[0], seq_name),
                'gt_path': '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s/groundtruth.txt' % (seq_name.split('-')[0], seq_name),
                'start_frame': None,  # 起始帧，默认为空
                'end_frame': None,
                'interval': 1,  # 绘制间隔
                'baseline1': '/home/wyp/project/SOITrack-main/output/test/tracking_results/odtrack/baseline_300/lasot/%s.txt' % seq_name,
                'baseline2': '/home/wyp/project/SOITrack-main/output/test/soi_tracking_results/odtrack/baseline/lasot/%s.txt' % seq_name,
                # 'baseline3': '/mnt/second/wangyipei/SOI/tracker_result/OSTrack/lasotSOI/test/results/OSTrack/%s.txt' % seq_name,
                # baseline4用黄色
            },
    }

    thickness = 2

    for key, val in demo_seqs.items():
        save_seq_dir = os.path.join(save_dir, key, '%s' % seq_name)
        makedir(save_seq_dir)

        filenames = read_filename(val['image_path'])

        gts = np.loadtxt(val['gt_path'], delimiter=',')

        # baseline1s = np.loadtxt(val['baseline1'], delimiter='\t')
        baseline1s = np.loadtxt(val['baseline1'], delimiter='\t')
        baseline2s = np.loadtxt(val['baseline2'], delimiter='\t')
        # baseline3s = np.loadtxt(val['baseline3'], delimiter=',')

        # if start_frame != None or end_frame != None:
        #     gts = gts[start_frame:end_frame]
        #     baseline1s = baseline1s[start_frame:end_frame]
        #     baseline2s = baseline2s[start_frame:end_frame]
        #     baseline3s = baseline3s[start_frame:end_frame]
        #     filenames = filenames[start_frame:end_frame]

        print(len(gts), len(filenames), len(baseline1s), len(baseline2s))

        assert (len(gts) == len(filenames) == len(baseline1s) == len(baseline2s))

        i = 0
        while i < len(filenames):
            filename = filenames[i]

            name = filename.split('/')[-1]
            print(name)

            gt = [int(num) for num in gts[i]]
            baseline1 = [int(num) for num in baseline1s[i]]
            baseline2 = [int(num) for num in baseline2s[i]]
            # baseline3 = [int(num) for num in baseline3s[i]]
            # baseline4 = [int(num) for num in baseline4s[i]]

            img = cv.imread(filename)

            cv.rectangle(img, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 255, 0), thickness)  # gt用绿色
            cv.rectangle(img, (baseline1[0], baseline1[1]), (baseline1[2] + baseline1[0], baseline1[3] + baseline1[1]),
                         (255, 0, 0), thickness)  # baseline1用蓝色
            cv.rectangle(img, (baseline2[0], baseline2[1]), (baseline2[2] + baseline2[0], baseline2[3] + baseline2[1]),
                         (0, 0, 255), thickness)  # baseline2用红色
            # cv.rectangle(img, (baseline3[0], baseline3[1]), (baseline3[2] + baseline3[0], baseline3[3] + baseline3[1]),
                        #  (0, 255, 255), thickness)  # baseline3用黄色
            # cv.rectangle(img, (baseline4[0], baseline4[1]), (baseline4[2] + baseline4[0], baseline4[3] + baseline4[1]),
            #              (255, 0, 255), thickness)  # baseline4用紫色

            cv.imwrite(os.path.join(save_seq_dir, name), img)

            i += int(val['interval'])


def draw_got10k_demo(seq_name):
    """绘制demo"""
    save_dir = '/mnt/second/wangyipei/SOI/demo'
    # 改每个baseline里面的路径以及序列名字就行
    demo_seqs = {
        'got10k':
            {
                'image_path': '/mnt/first/hushiyu/SOT/GOT-10k/data/val/%s' % seq_name,
                'gt_path': '/mnt/first/hushiyu/SOT/GOT-10k/data/val/%s/groundtruth.txt' % seq_name,
                'start_frame': None,  # 起始帧，默认为空
                'end_frame': None,
                'interval': 1,  # 绘制间隔
                'baseline1': '/mnt/second/wangyipei/SOI/tracker_result/transKT/got10kSOI/val/results/%s/%s_001.txt' % (
                    seq_name, seq_name),  # baseline1用红色
                'baseline2': '/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/got10kSOI/val/results/KeepTrack/%s.txt' %
                             seq_name.split('_')[-1],  # baseline2用蓝色
                'baseline3': '/mnt/second/wangyipei/SOI/tracker_result/ToMP/got10kSOI/val/results/ToMP/%s.txt' % seq_name,
                # baseline3用黄色
                'baseline4': '/mnt/second/wangyipei/SOI/tracker_result/OSTrack/got10kSOI/val/results/OSTrack/%s.txt' % seq_name,
                # baseline4用紫色
            },
    }

    thickness = 2

    for key, val in demo_seqs.items():
        save_seq_dir = os.path.join(save_dir, key, '%s' % seq_name)
        makedir(save_seq_dir)

        filenames = read_filename(val['image_path'])

        gts = np.loadtxt(val['gt_path'], delimiter=',')

        baseline1s = np.loadtxt(val['baseline1'], delimiter=',')
        baseline2s = np.loadtxt(val['baseline2'], delimiter=',')
        baseline3s = np.loadtxt(val['baseline3'], delimiter='\t')
        baseline4s = np.loadtxt(val['baseline4'], delimiter='\t')

        # if start_frame != None or end_frame != None:
        #     gts = gts[start_frame:end_frame]
        #     baseline1s = baseline1s[start_frame:end_frame]
        #     baseline2s = baseline2s[start_frame:end_frame]
        #     baseline3s = baseline3s[start_frame:end_frame]
        #     filenames = filenames[start_frame:end_frame]

        print(len(gts), len(filenames), len(baseline1s), len(baseline2s), len(baseline3s))

        assert (len(gts) == len(filenames) == len(baseline1s) == len(baseline2s) == len(baseline3s))

        i = 0
        while i < len(filenames):
            filename = filenames[i]

            name = filename.split('/')[-1]
            print(name)

            gt = [int(num) for num in gts[i]]
            baseline1 = [int(num) for num in baseline1s[i]]
            baseline2 = [int(num) for num in baseline2s[i]]
            baseline3 = [int(num) for num in baseline3s[i]]
            baseline4 = [int(num) for num in baseline4s[i]]

            img = cv.imread(filename)

            cv.rectangle(img, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 255, 0), thickness)  # gt用绿色
            cv.rectangle(img, (baseline1[0], baseline1[1]), (baseline1[2] + baseline1[0], baseline1[3] + baseline1[1]),
                         (0, 0, 255), thickness)  # baseline1用红色
            cv.rectangle(img, (baseline2[0], baseline2[1]), (baseline2[2] + baseline2[0], baseline2[3] + baseline2[1]),
                         (255, 0, 0), thickness)  # baseline2用蓝色
            cv.rectangle(img, (baseline3[0], baseline3[1]), (baseline3[2] + baseline3[0], baseline3[3] + baseline3[1]),
                         (0, 255, 255), thickness)  # baseline3用黄色
            cv.rectangle(img, (baseline4[0], baseline4[1]), (baseline4[2] + baseline4[0], baseline4[3] + baseline4[1]),
                         (255, 0, 255), thickness)  # baseline4用紫色

            cv.imwrite(os.path.join(save_seq_dir, name), img)

            i += int(val['interval'])


def draw_vot_demo(seq_name):
    """绘制demo"""
    save_dir = '/mnt/second/wangyipei/SOI/demo'

    def _corner2rect(corners, center=False):
        cx = np.mean(corners[:, 0::2], axis=1)
        cy = np.mean(corners[:, 1::2], axis=1)

        x1 = np.min(corners[:, 0::2], axis=1)
        x2 = np.max(corners[:, 0::2], axis=1)
        y1 = np.min(corners[:, 1::2], axis=1)
        y2 = np.max(corners[:, 1::2], axis=1)

        area1 = np.linalg.norm(corners[:, 0:2] - corners[:, 2:4], axis=1) * \
                np.linalg.norm(corners[:, 2:4] - corners[:, 4:6], axis=1)
        area2 = (x2 - x1) * (y2 - y1)
        scale = np.sqrt(area1 / area2)
        w = scale * (x2 - x1) + 1
        h = scale * (y2 - y1) + 1

        if center:
            return np.array([cx, cy, w, h]).T
        else:
            return np.array([cx - w / 2, cy - h / 2, w, h]).T

    # 改每个baseline里面的路径以及序列名字就行
    demo_seqs = {
        'VOTLT19':
            {
                'image_path': '/mnt/first/hushiyu/SOT/VOTLT2019/data/%s/color' % seq_name,
                'gt_path': '/mnt/first/hushiyu/SOT/VOTLT2019/data/%s/groundtruth.txt' % seq_name,
                'start_frame': None,  # 起始帧，默认为空
                'end_frame': None,
                'interval': 1,  # 绘制间隔
                'baseline1': '/mnt/second/wangyipei/SOI/tracker_result/TransKT/votSOI/test/VOTLT2019/TransKTtune3/%s.txt' % seq_name,
                # baseline1用红色
                'baseline2': '/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/votSOI/test/VOTLT2019/KeepTrack/%s.txt' % seq_name,
                # baseline2用蓝色
                'baseline3': '/mnt/second/wangyipei/SOI/tracker_result/ToMP/votSOI/test/VOTLT2019/ToMP/%s.txt' % seq_name,
                # baseline3用黄色
                'baseline4': '/mnt/second/wangyipei/SOI/tracker_result/OSTrack/votSOI/test/VOTLT2019/OSTrack/%s.txt' % seq_name,
                # baseline4用紫色
            },
    }

    thickness = 2

    for key, val in demo_seqs.items():
        save_seq_dir = os.path.join(save_dir, key, '%s' % seq_name)
        if os.path.exists(save_seq_dir):
            continue
        makedir(save_seq_dir)

        filenames = read_filename(val['image_path'])

        gts = np.loadtxt(val['gt_path'], delimiter=',')
        if gts.shape[1] == 8:
            gts = _corner2rect(gts)
        gts = np.where(np.isnan(gts), 0, gts)
        baseline1s = np.loadtxt(val['baseline1'], delimiter=',')
        baseline2s = np.loadtxt(val['baseline2'], delimiter=',')
        baseline3s = np.loadtxt(val['baseline3'], delimiter=',')
        baseline4s = np.loadtxt(val['baseline4'], delimiter=',')

        # if start_frame != None or end_frame != None:
        #     gts = gts[start_frame:end_frame]
        #     baseline1s = baseline1s[start_frame:end_frame]
        #     baseline2s = baseline2s[start_frame:end_frame]
        #     baseline3s = baseline3s[start_frame:end_frame]
        #     filenames = filenames[start_frame:end_frame]

        print(len(gts), len(filenames), len(baseline1s), len(baseline2s), len(baseline3s))

        assert (len(gts) == len(filenames) == len(baseline1s) == len(baseline2s) == len(baseline3s))

        i = 0
        while i < len(filenames):
            filename = filenames[i]
            name = filename.split('/')[-1]
            # print(name)

            gt = [int(num) for num in gts[i]]
            baseline1 = [int(num) for num in baseline1s[i]]
            baseline2 = [int(num) for num in baseline2s[i]]
            baseline3 = [int(num) for num in baseline3s[i]]
            baseline4 = [int(num) for num in baseline4s[i]]

            img = cv.imread(filename)

            cv.rectangle(img, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]), (0, 255, 0), thickness)  # gt用绿色
            cv.rectangle(img, (baseline1[0], baseline1[1]), (baseline1[2] + baseline1[0], baseline1[3] + baseline1[1]),
                         (0, 0, 255), thickness)  # baseline1用红色
            cv.rectangle(img, (baseline2[0], baseline2[1]), (baseline2[2] + baseline2[0], baseline2[3] + baseline2[1]),
                         (255, 0, 0), thickness)  # baseline2用蓝色
            cv.rectangle(img, (baseline3[0], baseline3[1]), (baseline3[2] + baseline3[0], baseline3[3] + baseline3[1]),
                         (0, 255, 255), thickness)  # baseline3用黄色
            cv.rectangle(img, (baseline4[0], baseline4[1]), (baseline4[2] + baseline4[0], baseline4[3] + baseline4[1]),
                         (255, 0, 255), thickness)  # baseline4用紫色

            cv.imwrite(os.path.join(save_seq_dir, name), img)

            i += int(val['interval'])


if __name__ == '__main__':
    # lasot_seq_name = 'sheep-9'
    # got10k_seq_name = 'GOT-10k_Val_000056'
    lasot_seq_names = [
        "sheep-9",
        "tiger-6",
        "turtle-8",
        "bird-2",
        "robot-1",
        "tiger-4",
        "hat-1",
        "hand-9",
        "zebra-10",
        "umbrella-9",
        "volleyball-19",
        "person-5",
        "kite-10",
        "pig-2",
        "robot-5",
        "shark-2",
        "zebra-16",
        "umbrella-19",
        "monkey-9",
        "goldfish-7",
        "microphone-2",
        "sheep-3",
        "yoyo-7"
    ]
    for lasot_seq_name in lasot_seq_names:
        # print(lasot_seq_name)
        draw_lasot_demo(seq_name=lasot_seq_name)
        print('done')
    # with open('/mnt/second/wangyipei/SOI/data/VOTLT2019/demo.txt') as f:
    #     vot_seq_names = f.read().strip().split('\n')
    # for vot_seq_name in vot_seq_names:
    #     print(vot_seq_name)
    #     draw_vot_demo(seq_name=vot_seq_name)
    #     print('done')
    # draw_got10k_demo(seq_name=got10k_seq_name)
    # for videocube_seq_name in videocube_seq_names:
    #     print(videocube_seq_name)
    #     draw_videocube_demo(seq_name=videocube_seq_name)
    #     print('done')
