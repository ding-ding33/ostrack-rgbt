class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/pretrained_networks'
        self.got10k_val_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/got10k/val'
        self.lasot_lmdb_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/coco_lmdb'
        self.coco_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/coco'
        self.lasot_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/lasot'
        self.got10k_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/got10k/train'
        self.trackingnet_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/trackingnet'
        self.depthtrack_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt/ceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/depthtrack/train'
        self.lasher_dir = '/data1/Datasets/Tracking/LasHeR/'
        self.visevent_dir = '/data1/Code/dongzhaodong/dingceyu/ostrack-rgbt//New_work/ostrack_rgbt/data/visevent/train'
