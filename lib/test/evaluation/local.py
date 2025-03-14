'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-12-12 15:10:22
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-12-12 15:11:57
FilePath: /SOITrack/lib/test/evaluation/local.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/first/hushiyu/SOT/GOT-10k/data/got10k_lmdb'
    settings.got10k_path = '/mnt/first/hushiyu/SOT/GOT-10k/data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/itb'
    settings.lasot_extension_subset_path_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/lasot_lmdb'
    settings.lasot_path = '/mnt/first/hushiyu/SOT/LaSOT/data'
    settings.network_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/nfs'
    settings.otb_path = '/mnt/first/hushiyu/SOT/OTB/data'
    settings.prj_dir = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack'
    settings.result_plot_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/output/test/result_plots'
    settings.results_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/output'
    settings.segmentation_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/output/test/segmentation_results'
    settings.tc128_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/second/wangyipei/trackingnet'
    settings.uav_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/uav'
    settings.vot18_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/vot2018'
    settings.vot22_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/vot2022'
    settings.vot_path = '/seu_nvme/home/luxiaobo/230248984/code/SOITrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

