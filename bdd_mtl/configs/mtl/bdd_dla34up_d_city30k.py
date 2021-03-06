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

cfg = get_configs('dla34up-d')

# Override default configs. Feel free to override more fields
cfg.optimizer['lr'] = 0.02
cfg.lr_config['step'] = [16, 22]
cfg.total_epochs = 24
cfg.data['imgs_per_gpu'] = 4
cfg.data['workers_per_gpu'] = 4
cfg.work_dir = './work_dirs/debug/BDD-d_city-30k/dla34up'
cfg.load_from = None
cfg.resume_from = None
cfg.data['train'][0]['ann_file'] = cfg.data['train'][0]['ann_file'][:-5] + '_city_30k.json'
cfg.data['test'][0]['ann_file'] = cfg.data['val'][0]['ann_file'][:-5] + '_city.json'

for k, v in cfg.__dict__.items():
    if not k.startswith('__'):
        setattr(sys.modules[__name__], k, v)
