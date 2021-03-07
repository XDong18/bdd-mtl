export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29503 ./tools/dist_train.sh \
cbdd_mtl/configs/new_config/dla34_tagging.py 4