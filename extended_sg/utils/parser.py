import yaml
import os
import argparse
from pathlib import Path
from argparse import ArgumentParser
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from conf import Config

class Parser():
    def __init__(self, config_path:str="./config/config.yaml"):
        
        self.parser = ArgumentParser(description="Extract extended scene graphs from images' identikit.", formatter_class=argparse.RawTextHelpFormatter)
        self.config_path = config_path

        self.parser.add_argument('-c', '--config', type=str, required=True if config_path is None else False, default=config_path, dest='CONFIG', help='Configuration file path')
        
        self.parser.add_argument('-mor', '--min-obj-rel', type=int, dest="MOR", help="Overrides set minimum objects to have relations")
        self.parser.add_argument('-bm', '--bbox-match', type=float, dest="BM", help="Overrides set bounding box matching score")
        self.parser.add_argument('-cc', '--compl-cont', type=float, dest="CC", help="Overrides set percentage of pixels contained in a bounding box")
        self.parser.add_argument('-do', '--depth-overlap', type=float, dest="DO", help="Overrides set overlap for depth-driven relations")
        self.parser.add_argument('-dul', '--depth-up-lim', type=float, dest="DUL", help="Overrides set depth upper limit")
        self.parser.add_argument('-dll', '--depth-low-lim', type=float, dest="DLL", help="Overrides set depth lower limit")
        self.parser.add_argument('-prt', '--pos-rel-tol', type=float, dest="PRT", help="Overrides set positional relation tolerance")

        self.parser.add_argument('-id', '--input-dir', type=str, dest="ID", help="Overrides set identikits input directory")
        self.parser.add_argument('-dd', '--depth-dir', type=str, dest="DD", help="Overrides set depth maps input directory")
        self.parser.add_argument('-pd', '--panopt-dir', type=str, dest="PD", help="Overrides set panoptic maps input directory")
        self.parser.add_argument('-sgd', '--scene-graph-dir', type=str, dest="SGD", help="Overrides set scene graph input directory")
        self.parser.add_argument('-od', '--output-dir', type=str, dest="OD", help="Overrides set extended scene graphs output directory")

        self.parser.add_argument('-it', '--input-type', type=str, dest="IT", help="Overrides set identikits filetype")
        self.parser.add_argument('-dt', '--depth-type', type=str, dest="DT", help="Overrides set depth maps filetype")
        self.parser.add_argument('-pt', '--panopt-type', type=str, dest="PT", help="Overrides set panoptic maps filetype")
        self.parser.add_argument('-sgt', '--scene-graph-type', type=str, dest="SGT", help="Overrides set scene graph filetype")
        self.parser.add_argument('-ot', '--output-type', type=str, dest="OT", help="Overrides set extended scene graphs filetype")

    def parse_args(self):
        self.args = self.parser.parse_args()

        #load yaml file to the configuration
        with open(self.args.CONFIG, "r") as file:
            d = yaml.safe_load(file)
            config = Config(**d)

        if self.args.MOR is not None:
            config.threshold.min_objects_for_relations = self.args.MOR
        if self.args.BM is not None:
            config.threshold.bbox_match = self.args.BM
        if self.args.CC is not None:
            config.threshold.completely_contained = self.args.CC
        if self.args.DO is not None:
            config.threshold.depth_overlap = self.args.DO
        if self.args.DUL is not None:
            config.threshold.depth_upper_limit = self.args.DUL
        if self.args.DLL is not None:
            config.threshold.depth_lower_limit = self.args.DLL
        if self.args.PRT is not None:
            config.threshold.positional_relation_tolerance = self.args.PRT

        if self.args.ID is not None:
            config.filesystem.input = self.args.ID
        if self.args.DD is not None:
            config.filesystem.depth = self.args.DD
        if self.args.PD is not None:
            config.filesystem.panoptic = self.args.PD
        if self.args.SGD is not None:
            config.filesystem.scene_graph = self.args.SGD
        if self.args.OD is not None:
            config.filesystem.output = self.args.OD

        if self.args.IT is not None:
            config.filetype.input = self.args.IT
        if self.args.DT is not None:
            config.filetype.depth = self.args.DT
        if self.args.PT is not None:
            config.filetype.panoptic = self.args.PT
        if self.args.SGT is not None:
            config.filetype.scene_graph = self.args.SGT
        if self.args.OT is not None:
            config.filetype.output = self.args.OT

        return config