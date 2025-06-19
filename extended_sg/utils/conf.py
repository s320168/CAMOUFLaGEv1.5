from pydantic import BaseModel

class Threshold(BaseModel):
    # make it possible to have relationonly when having at least this amount of objects
    min_objects_for_relations: int
    # iou lower limit to assert two matching bounding boxes
    bbox_match: float
    # percentage of pixels of a mask that a bounding box must include to be considered completely contained
    completely_contained: float
    # iou lower limit to consider two bounding boxes depth relation
    depth_overlap: float
    # limit over which consider depth ratio between objects as relevant
    depth_upper_limit: float
    # limit under which consider depth ratio between objects as relevant
    depth_lower_limit: float
    # lower limit to consider relative position between objects as relevant
    positional_relation_tolerance: float

class Filesystem(BaseModel): # paths containing the used data
    # directory containing identikits produced by FRESCO
    input: str
    # directory containing depth estimation of the images
    depth: str
    # directory containing panoptic segmentation of the images
    panoptic: str
    # directory containing scene graphs of the images
    scene_graph: str
    # directory containing the extended scene graphs
    output: str

class Filetype(BaseModel): # file type of data used
    # input identikit file type
    input: str
    # depth file type
    depth: str
    # panoptic segmentation file type
    panoptic: str
    # scene graph file type
    scene_graph: str
    # output extended scene graph file type
    output: str

class Config(BaseModel):
    threshold: Threshold
    filesystem: Filesystem
    filetype: Filetype