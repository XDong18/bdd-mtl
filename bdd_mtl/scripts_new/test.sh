export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29503 ./tools/dist_test.sh configs/retinanet/retinanet_r101_fpn_1x_bdd100k.py \
    b_out/c_out/retinanet_r101_fpn_1x/latest.pth \
    4 --eval bbox --val_dataset