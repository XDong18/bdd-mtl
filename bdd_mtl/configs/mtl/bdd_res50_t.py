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
cfg.optimizer['lr'] = 0.04
cfg.lr_config['step'] = [4, 5]
cfg.total_epochs = 6
cfg.data['imgs_per_gpu'] = 8
cfg.data['workers_per_gpu'] = 8
cfg.work_dir = './work_dirs/debug/BDD-t/res50'
cfg.load_from = None
cfg.resume_from = None

for k, v in cfg.__dict__.items():
    if not k.startswith('__'):
        setattr(sys.modules[__name__], k, v)
