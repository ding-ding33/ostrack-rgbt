from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/got10k_lmdb'
    settings.got10k_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/itb'
    settings.lasot_extension_subset_path_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/lasot_lmdb'
    settings.lasot_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/lasot'
    settings.network_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/nfs'
    settings.otb_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/otb'
    settings.prj_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt'
    settings.result_plot_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/output/test/result_plots'
    settings.results_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/output'
    settings.segmentation_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/output/test/segmentation_results'
    settings.tc128_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/trackingnet'
    settings.uav_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/uav'
    settings.vot18_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/vot2018'
    settings.vot22_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/vot2022'
    settings.vot_path = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

