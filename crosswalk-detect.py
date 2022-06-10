#!/usr/bin/env python3
import tempfile
import torch
import subprocess
import json
import argparse
from pathlib import PosixPath

from track import detect

YOLO_MODEL = 'ripo-v04.2.pt'
DEEP_SORT_MODEL = 'osnet_ibn_x1_0_MSMT17'


def run_detector(args):
    with torch.no_grad():
        detect(argparse.Namespace(
            yolo_model=YOLO_MODEL,
            deep_sort_model=DEEP_SORT_MODEL,
            source=args.input,
            output='inference/output',
            imgsz=[1088, 1088],
            conf_thres=0.5,
            iou_thres=0.5,
            fourcc='mp4v',
            device='',
            show_vid=True,
            save_vid=args.save_vid,
            save_txt=False,
            classes=[0],
            agnostic_nms=False,
            augment=False,
            update=False,
            evaluate=False,
            config_deepsort='deep_sort/configs/deep_sort.yaml',
            half=False,
            visualize=False,
            max_det=1000,
            save_crop=False,
            dnn=False,
            project=PosixPath(args.output_dir),
            name='exp',
            exist_ok=True,
        ))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='input video path',
        required=True
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='output directory',
        required=True
    )
    parser.add_argument(
        '--save-vid',
        action='store_true',
        help='save video tracking results'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run_detector(args)
