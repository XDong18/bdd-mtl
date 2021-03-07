from bdd_mtl_factory import get_configs
import sys

################################################################################
#
# Format of the command to get configs:
#
#   $MODEL_NAME-$TASKS
#
#   task correspondence:
#     l - Lane marking
#     r - Drivable area
#     s - Semantic segmentation
#     d - Detection
#     i - Instance Segmentation
#     t - Multiple object Tracking
#     x - Multiple object Tracking with Segmentation
#
################################################################################
class BddMtlConfig:
    def __init__(self):
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.model = dict(
            type='MTL',
            pretrained='/shared/xudongliu/code/weights/dla34-ba72cf86.pth',
            backbone=dict(
                type='DLA',
                levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block_num=2,
                return_levels=True)
        )
        self.train_cfg = dict()
        self.test_cfg = dict()
        self.dataset_type = []
        self.data_root = '/shared/xudongliu/bdd100k/'
        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.data = dict(
            imgs_per_gpu=2,
            workers_per_gpu=2,
            train=[],
            val=[],
            test=[])
        # optimizer
        self.optimizer = dict(
            type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) # , track_enhance=10)
        self.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
        # learning policy
        self.lr_config = dict(
            policy='step',
            warmup='exp',
            warmup_iters=500,
            warmup_ratio=0.1 / 3,
            step=[8, 11])
        self.checkpoint_config = dict(interval=1)
        # yapf:disable
        self.log_config = dict(
            interval=10,
            hooks=[
                dict(type='WandBLoggerHook', project_name='bdd-'),
                dict(type='TextLoggerHook')
                # dict(type='TensorboardLoggerHook')
            ])
        # yapf:enable
        # runtime settings
        self.total_epochs = 12
        self.dist_params = dict(backend='nccl')
        self.log_level = 'INFO'
        self.load_from = None
        self.work_dir = './work_dirs/debug/bdd'
        self.resume_from = None
        self.workflow = [('train', 1)]

# model 
cfg = BddMtlConfig()
cfg.model.update(dict(
        cls_head=dict(
            type='ClsHead',
            num_convs=2,
            in_channels=512,
            conv_kernel_size=3,
            conv_out_channels=256,
            num_classes=[6, 6, 3],
            conv_cfg=None,
            norm_cfg=None,
            loss_cls=dict(
                type='CrossEntropyLoss',
                ignore_index=-1)
        )
    ))

# dataset
cfg.dataset_type.append(['BddCls'])
    cfg.data['train'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/train/',
            label_dir=cfg.data_root+'labels/cls/cls_train.json',
            phase='train',
            flip_ratio=0.5,
            img_prefix=cfg.data_root+'images/100k/train/',
            img_scale=(1280, 720),
            crop_size=(640, 640),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            ann_file=None
        )
    )
    cfg.data['val'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/cls/cls_val.json',
            phase='val',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            test_mode=True,
            ann_file=None
        )
    )
    cfg.data['test'].append(
        dict(type='BddCls',
            image_dir=cfg.data_root+'images/100k/val/',
            label_dir=cfg.data_root+'labels/cls/cls_val.json',
            phase='test',
            flip_ratio=0,
            img_prefix=cfg.data_root+'images/100k/val/',
            img_scale=(1280, 720),
            img_norm_cfg=cfg.img_norm_cfg,
            size_divisor=32,
            test_mode=True,
            ann_file=None
        )
    )
    # output settings
    cfg.work_dir += 'c'
    cfg.log_config['hooks'][0]['project_name'] += 'c'

# Override default configs. Feel free to override more fields
cfg.optimizer['lr'] = 0.002
cfg.lr_config['step'] = [80, 110]
cfg.total_epochs = 120
cfg.data['imgs_per_gpu'] = 8
cfg.data['workers_per_gpu'] = 8
cfg.work_dir = './output/dla34_tagging/'
cfg.load_from = None
cfg.resume_from = None # './work_dirs/debug/BDD-c/dla34/latest.pth'

for k, v in cfg.__dict__.items():
    if not k.startswith('__'):
        setattr(sys.modules[__name__], k, v)
