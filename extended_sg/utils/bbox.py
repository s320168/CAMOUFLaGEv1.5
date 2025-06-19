import numpy as np
from copy import copy

class BoundingBox():
    def __init__(self, *args):
        # if constructor receives a single argument which is a tuple
        if len(args) == 1 and type(args[0]) == dict:
            self.x0 = args[0]["x0"]
            self.y0 = args[0]["y0"]
            self.x1 = args[0]["x1"]
            self.y1 = args[0]["y1"]
        # if constructor receives all the arguments in separate variables
        elif len(args) == 4:
            self.x0 = args[0]
            self.y0 = args[1]
            self.x1 = args[2]
            self.y1 = args[3]
        self.instances_found = {}

    def get_instances_found(self) -> dict:
        # returns a dictionary containing the instance id and the corresponding pixel count
        return copy(self.instances_found)
    
    def get_coords(self) -> dict:
        # returns a dictionary containing the bounding box's coordinates
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
    
    def get_x0(self) -> int:
        return self.x0
    
    def get_y0(self) -> int:
        return self.y0
    
    def get_x1(self) -> int:
        return self.x1
    
    def get_y1(self) -> int:
        return self.y1
    
    def get_x_dimension(self) -> int:
        # returns bounding box's width (+1 since pixels start at (0, 0))
        return self.x1 - self.x0 + 1
    
    def get_y_dimension(self) -> int:
        # returns bounding box's height (+1 since pixels start at (0, 0))
        return self.y1 - self.y0 + 1
    
    def get_area(self) -> int:
        # compute area of a rectagle
        return self.get_x_dimension() * self.get_y_dimension()

    def get_instance_count(self, key) -> int:
        # return how many pixels of the selected instance have been found
        return self.instances_found[key]["count"]
    
    def get_instance_norm(self, key) -> float:
        # return how many pixels of the selected instance with respect to the box's area
        return self.instances_found[key]["norm_presence"]
    
    def get_instance_ratio(self, panoptic_map, key) -> float:
        # return how many pixels of the selected instance with respect to the whole image
        total_instances = np.where(panoptic_map == key, 1, 0).sum()
        return self.instances_found[key]["count"]/total_instances if total_instances > 0 else 0

    def compute_instances_norm(self, panoptic_map, max_id) -> None:
        # compute the percentage of pixels in the bounding box which belong to each instance
        for i in range(1, max_id+1):
            # count how many pixels belong to the i-th instance
            count = np.where(panoptic_map[self.y0:self.y1+1, self.x0:self.x1+1] == i, 1, 0).sum()
            # assign only counts > 0 while skipping the others
            if count > 0:
                self.instances_found[i] = {"count": count}        
        for key in self.instances_found.keys():
            # divide the pixel count by the total amount of pixel in the box
            self.instances_found[key]["norm_presence"] = float(self.instances_found[key]["count"]/self.get_area())

    def contains_point(self, x, y) -> bool:
        # checks if a box contains a (x, y) point in its area (not including the sides)
        return self.x0 < x and self.x1 > x and self.y0 < y and self.y1 > y
    
def iou(boxA, boxB) -> float:
    # determine the coordinates of the intersection rectangle
    x_left = max(boxA.get_x0(), boxB.get_x0())
    y_top = max(boxA.get_y0(), boxB.get_y0())
    x_right = min(boxA.get_x1(), boxB.get_x1())
    y_bottom = min(boxA.get_y1(), boxB.get_y1())

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(boxA.get_area() + boxB.get_area() - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def x_axis_distance(boxA, boxB) -> int:
    # returns an integer related on how many pixels the bounding boxes have between their centroids 
    x_centroid_A = (boxA.get_x1() + boxA.get_x0())/2
    x_centroid_B = (boxB.get_x1() + boxB.get_x0())/2
    # > 0: boxB is to the right of boxA, < 0: boxB is to the left of boxA
    return x_centroid_A - x_centroid_B

def y_axis_distance(boxA, boxB) -> int:
    # returns an integer related on how many pixels the bounding boxes have between their centroids 
    y_centroid_A = (boxA.get_y1() + boxA.get_y0())/2
    y_centroid_B = (boxB.get_y1() + boxB.get_y0())/2
    # > 0: boxB is below boxA, < 0: boxB is above boxA
    return y_centroid_A - y_centroid_B

def A_contains_B(boxA, boxB) -> bool:
    # returns True when other's coordinates are contained in self
    return boxA.contains_point(boxB.get_x0(), boxB.get_y0()) and boxA.contains_point(boxB.get_x1(), boxB.get_y1())