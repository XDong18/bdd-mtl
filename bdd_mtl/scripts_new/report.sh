export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_SERVICE_FORCE_INTEL=1
python report_info.py b_out/cornernet_hourglass104_mstest_4x6_210e/latest.pth --out_name mmdetection-cornetnet_hourglass104-210e --config configs/cornernet/cornernet_hourglass104_mstest_4x6_210e_bdd100k.py