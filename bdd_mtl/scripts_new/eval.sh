export CUDA_VISIBLE_DEVICES=8,9

PORT=29502 ./tools/dist_test.sh configs/cornernet/cornernet_hourglass104_mstest_4x6_210e_bdd100k.py out/cornernet_hourglass104_mstest_4x6_210e/latest.pth 2 --eval bbox --val_dataset --eval-options "classwise=True"