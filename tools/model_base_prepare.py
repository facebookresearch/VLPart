# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='output/fpn_coco_base/model_final.pth')
    parser.add_argument('--output_path', default='models/fpn_coco_base.pth')
    args = parser.parse_args()
    
    obj = torch.load(args.base_path, map_location="cpu")
    obj.pop("optimizer")
    obj.pop("scheduler")
    obj.pop("iteration")
    
    torch.save(obj, args.output_path)
    