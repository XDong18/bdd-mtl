export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG=$1
GPUS=$2
FOLDER=work_dirs

python3 -m torch.distributed.launch --nproc_per_node=$GPUS ⁠\
./tools/train.py \
./configs/${CONFIG}.py \
--work_dir=./${FOLDER}/${CONFIG} \
--launcher pytorch
