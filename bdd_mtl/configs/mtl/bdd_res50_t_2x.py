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

cfg = get_configs('res50-t')

# Override default configs. Feel free to override more fields
cfg.optimizer['lr'] = 0.02
cfg.lr_config['step'] = [8, 10]
cfg.total_epochs = 12
cfg.data['imgs_per_gpu'] = 8
cfg.data['workers_per_gpu'] = 4
cfg.work_dir = './work_dirs/debug/BDD-t/res50_2x'
cfg.load_from = None
cfg.resume_from = './work_dirs/debug/BDD-t/res50/epoch_3.pth'

for k, v in cfg.__dict__.items():
    if not k.startswith('__'):
        setattr(sys.modules[__name__], k, v)
