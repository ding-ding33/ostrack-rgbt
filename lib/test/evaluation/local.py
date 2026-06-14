from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    prj_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt'

    settings.prj_dir = prj_dir
    settings.save_dir = '{}/output'.format(prj_dir)

    settings.results_path = '{}/output/test/tracking_results'.format(prj_dir)
    settings.segmentation_path = '{}/output/test/segmentation_results'.format(prj_dir)
    settings.network_path = '{}/output/test/networks'.format(prj_dir)
    settings.result_plot_path = '{}/output/test/result_plots'.format(prj_dir)

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.lasher_dir = '/data1/Datasets/Tracking/LasHeR/'
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
