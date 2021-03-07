export CUDA_VISIBLE_DEVICES=0,1
export MKL_SERVICE_FORCE_INTEL=1
python report_info.py out/cascade_rcnn_r50_fpn_1x/cascade_rcnn_r50_fpn_1x-b915cc22.pth --out_name mmdetection-cascade_rcnn_r50_fpn-1x --config configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_bdd100k.py
python report_info.py out/cascade_rcnn_r101_fpn_1x/cascade_rcnn_r101_fpn_1x-8f6ef242.pth --out_name mmdetection-cascade_rcnn_r101_fpn-1x --config configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_bdd100k.py
python report_info.py out/cascade_rcnn_x101_32x4d_fpn_1x/cascade_rcnn_x101_32x4d_fpn_1x-46de620b.pth --out_name mmdetection-cascade_rcnn_x101_32x4d_fpn-1x --config configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_bdd100k.py
python report_info.py b_out/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x-99d59f90.pth --out_name mmdetection-fcos_center-normbbox-centeronreg-giou_r50_fpn_gn-head_dcn_4x4-1x --config configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_bdd100k.py
python report_info.py b_out/fcos_center-normbbox-centeronreg-giou_r101_caffe_fpn_gn-head_dcn_4x4_1x/fcos_center-normbbox-centeronreg-giou_r101_caffe_fpn_gn-head_dcn_4x4_1x-9167d889.pth --out_name mmdetection-fcos_center-normbbox-centeronreg-giou_r101_fpn_gn-head_dcn_4x4-1x --config configs/fcos/fcos_center-normbbox-centeronreg-giou_r101_caffe_fpn_gn-head_dcn_4x4_1x_bdd100k.py
python report_info.py out/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x-fcfb538d.pth --out_name mmdetection-faster_rcnn_r50_fpn_mdconv_c3-c5_group4-1x --config configs/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_bdd100k.py
python report_info.py b_out/faster_rcnn_r101_fpn_dconv_c3-c5_1x/latest.pth --out_name mmdetection-faster_rcnn_r101_fpn_dconv_c3-c5-1x --config configs/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_bdd100k.py
python report_info.py out/faster_rcnn_r50_pafpn_1x/latest.pth --out_name mmdetection-faster_rcnn_r50_pafpn-1x --config configs/pafpn/faster_rcnn_r50_pafpn_1x_bdd100k.py
python report_info.py out/faster_rcnn_r101_pafpn_1x/latest.pth --out_name mmdetection-faster_rcnn_r101_pafpn-1x --config configs/pafpn/faster_rcnn_r101_pafpn_1x_bdd100k.py

