class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/pretrained_networks'
        self.lasot_dir = '/mnt/first/hushiyu/SOT/LaSOT/data'
        self.got10k_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data/train'
        self.got10k_val_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data/val'
        self.lasot_lmdb_dir = '/home/micros/projects/OSTrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/micros/projects/OSTrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/mnt/second/wangyipei/trackingnet'
        self.trackingnet_lmdb_dir = '/home/micros/projects/OSTrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/second/wangyipei/coco_root'
        self.coco_lmdb_dir = '/home/micros/projects/OSTrack-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/micros/projects/OSTrack-main/data/vid'
        self.imagenet_lmdb_dir = '/home/micros/projects/OSTrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = '/home/micros/projects/pytracking-master/pytracking/util_scripts/target_candidates_dataset_dimp_simple_super_dimp_simple.json'
        self.iccv_dataset_path = '/mnt/third/zdl/ICCV_COM/'
