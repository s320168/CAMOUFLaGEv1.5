from PIL import Image
import numpy as np
import json
from bbox import BoundingBox, iou, x_axis_distance, y_axis_distance, A_contains_B

def add_objects(input_data: dict, output_data: dict) -> None:
    # initialize id counter
    id = 0
    # list the objects detected in the picture (taken from PrismerObjectDet and PrismerDepth)
    objs = input_data["plastic_level"]["topological_categories"]["obj_positions"]
    for t, p in zip(objs["object_labels"], objs["object_boxes"]):
        # skip human face objects as we will add them later
        if t != "human face":
            # each object gets added to the list with corresponding id, label ("type") and xyxy-bounding box ("position")
            output_data["objects"].append({
                "id": id, 
                "type": t, 
                "position": {"x0": p[0], "y0": p[1], "x1": p[2], "y1": p[3]}
            })
            # increment object id
            id += 1
    # add "human face" objects (taken from FACER)
    faces = input_data["figurative_level"]["content_participants"]["single_person_face_attributes"]
    for p, a in zip(faces["face_boxes"], faces["face_attributes_scores"]):
        output_data["objects"].append({
            "id": id, 
            "type": "human face", 
            "position": {"x0": p[0], "y0": p[1], "x1": p[2], "y1": p[3]}
        })
        # initialize attributes
        output_data["objects"][id]["depth"] = float()
        output_data["objects"][id]["attributes"] = dict()
        output_data["objects"][id]["attributes"] = {"face_attributes_scores": a}
        # increment object id
        id += 1


def add_objects_depth(input_data: dict, output_data: dict, input_filename: str, config: dict) -> None:
    # assignment of depth info to objects
    # depth map
    im_frame_d = Image.open(config.filesystem.depth + input_filename + config.filetype.depth)
    # panoptic segmentation map
    im_frame_p = Image.open(config.filesystem.panoptic + input_filename + '_panoptic' + config.filetype.panoptic)
    # assign maps to numpy arrays
    np_frame_d = np.array(im_frame_d.getdata()).reshape(im_frame_p.size[1], im_frame_p.size[0])
    np_frame_p = np.array(im_frame_p.getdata()).reshape(im_frame_p.size[1], im_frame_p.size[0])
    # store in a list every object's bounding box area
    objects_bbox = []
    for i in range(len(output_data["objects"])):
        bbox = BoundingBox(output_data["objects"][i]["position"])
        objects_bbox.append((i, bbox))
    # sort the bounding boxes by ascending areas
    objects_bbox.sort(key=lambda e: e[1].get_area())
    for b in objects_bbox:
        # compute how many pixels inside the bounding box correspond to every instance detected inside
        b[1].compute_instances_norm(np_frame_p, np_frame_p.max()+1)
        max_ratio = 0
        max_depth = 0
        # max_norm_presence = 0
        for i in b[1].get_instances_found():
            # keep track of the instance which has the highest ratio of pixels contained in the bounding box with respect to the instance's pixels across the entire image
            ratio = b[1].get_instance_ratio(np_frame_p, i)
            if ratio > max_ratio:
                max_ratio = ratio
                instance_max_ratio = i
            # keep track of the instance which is closer to the camera (it would probably be the most important instance in the bounding box)
            depth_data = np_frame_d[b[1].get_y0():b[1].get_y1()+1, b[1].get_x0():b[1].get_x1()+1] * np.where(np_frame_p[b[1].get_y0():b[1].get_y1()+1, b[1].get_x0():b[1].get_x1()+1] == i, 1, 0)
            mean_depth = depth_data.sum()/b[1].get_instance_count(i)
            if mean_depth > max_depth:
                max_depth = mean_depth
                instance_max_depth = i
        # if there is an instance almost completely or entirely contained in the bounding box: select it as the base used to get depth data
        if max_ratio > float(config.threshold.completely_contained):
            choice = instance_max_ratio
            depth_data = np_frame_d[b[1].get_y0():b[1].get_y1()+1, b[1].get_x0():b[1].get_x1()+1] * np.where(np_frame_p[b[1].get_y0():b[1].get_y1()+1, b[1].get_x0():b[1].get_x1()+1] == choice, 1, 0)
            # compute mean depth of the selected pixels
            depth = depth_data.sum()/b[1].get_instance_count(choice)
            # remove the chosen instance in order not to pick it later
            np_frame_p = np.where(np_frame_p == choice, -1, np_frame_p)
        # if no instance is almost completely contained in the bounding box, proceed on selecting the nearest surface to the camera
        else:
            choice = instance_max_depth
            depth = max_depth
        # assign depth data to object                
        output_data["objects"][b[0]]["depth"] = round(depth, 2)


def add_persons_characteristics(input_data: dict, output_data: dict, config: dict) -> None:
    # add persons characteristics (taken from DeepFace)
    persons = input_data["figurative_level"]["content_participants"]["single_person_characteristics"]
    # cycle every face box in persons
    for i in range(len(persons["face_boxes"])):
        # initialize current face's found flag
        found = False
        # create a BoundingBox object based on "face_boxes" info
        faceA_bbox = BoundingBox(persons["face_boxes"][i][0], persons["face_boxes"][i][1], persons["face_boxes"][i][2], persons["face_boxes"][i][3])
        # cycle every "human face" object
        for o in output_data["objects"]:
            if o["type"] != "human face":
                continue
            # create a BoundingBox object based on "human face" position attribute
            faceB_bbox = BoundingBox(o["position"]["x0"], o["position"]["y0"], o["position"]["x1"], o["position"]["y1"])
            # if the intersection over union of the created BoundingBox objects is greater than a certain percentage: assign its characteristics to the object
            if iou(faceB_bbox, faceA_bbox) > config.threshold.bbox_match:
                # set found flag
                found = True
                # add the new attributes
                o["attributes"]["age"] = persons["age"][i]
                o["attributes"]["gender_scores"] = persons["gender_scores"][i]
                o["attributes"]["ethnicity_scores"] = persons["ethnicity_scores"][i]
                break
        # if no object corresponds to the "face_boxes" info: create a new "human face" object with the available data
        if not found:
            new_face = dict()
            new_face["id"] = len(output_data["objects"])
            new_face["type"] = "human face"
            new_face["position"] = faceA_bbox.get_coords()
            new_face["attributes"] = {
                "age": persons["age"][i],
                "gender_scores": persons["gender_scores"][i],
                "ethnicity_scores": persons["ethnicity_scores"][i]
            }
            output_data["objects"].append(new_face)


def add_faces_emotions(input_data: dict, output_data: dict, config: dict) -> None:
    # add persons' emotions (taken from EmoNet)
    emotions = input_data["figurative_level"]["emotion"]["emotion"]
    # cycle every face box in faces
    for i in range(len(emotions["face_boxes"])):
        # initialize current face's found flag
        found = False
        # create a BoundingBox object based on "face_boxes" info
        faceA_bbox = BoundingBox(emotions["face_boxes"][i][0], emotions["face_boxes"][i][1], emotions["face_boxes"][i][2], emotions["face_boxes"][i][3])
        # cycle every "human face" object
        for o in output_data["objects"]:
            if o["type"] != "human face":
                continue
            # create a BoundingBox object based on "human face" position attribute
            faceB_bbox = BoundingBox(o["position"]["x0"], o["position"]["y0"], o["position"]["x1"], o["position"]["y1"])
            # if the intersection over union of the created BoundingBox objects is greater than a certain percentage: assign its characteristics to the object
            if iou(faceB_bbox, faceA_bbox) > config.threshold.bbox_match:
                # set found flag
                found = True
                # add the new attributes
                o["attributes"]["emotion_scores"] = emotions["emotion_scores"][i]
                break
        # if no object corresponds to the "face_boxes" info: create a new "human face" object with the available data
        if not found:
            new_face = dict()
            new_face["id"] = len(output_data["objects"])
            new_face["type"] = "human face"
            new_face["position"] = faceA_bbox.get_coords()
            new_face["attributes"] = {"emotion_scores": emotions["emotion_scores"][i]}
            output_data["objects"].append(new_face)


def add_head_pose(input_data: dict, output_data: dict, config: dict) -> None:
    # add heads pose (from 6DRepNet)
    head_poses = input_data["narrative_level"]["first_grade_secondary_watcher_looked_system"]["head_pose"]
    # cycle every face box in faces
    for i in range(len(head_poses["face_boxes"])):
        found = False
        # create a BoundingBox object based on "face_boxes" info
        faceA_bbox = BoundingBox(head_poses["face_boxes"][i][0], head_poses["face_boxes"][i][1], head_poses["face_boxes"][i][2], head_poses["face_boxes"][i][3])
        # cycle every "human face" object
        for o in output_data["objects"]:
            if o["type"] != "human face":
                continue
            # create a BoundingBox object based on "human face" position attribute
            faceB_bbox = BoundingBox(o["position"]["x0"], o["position"]["y0"], o["position"]["x1"], o["position"]["y1"])
            # if the intersection over union of the created BoundingBox objects is greater than a certain percentage: assign its characteristics to the object
            if iou(faceB_bbox, faceA_bbox) > config.threshold.bbox_match:
                # set found flag
                found = True
                # add the new attributes
                o["attributes"]["head_pose"] = {"yaw": head_poses["yaw"][i], "pitch": head_poses["pitch"][i], "roll": head_poses["roll"][i]}
                break
        # if no object corresponds to the "face_boxes" info: create a new "human face" object with the available data
        if not found:
            new_face = dict()
            new_face["id"] = len(output_data["objects"])
            new_face["type"] = "human face"
            new_face["position"] = faceA_bbox.get_coords()
            new_face["attributes"] = {"head_pose": {"yaw": head_poses["yaw"][i], "pitch": head_poses["pitch"][i], "roll": head_poses["roll"][i]}}
            output_data["objects"].append(new_face)


def add_gaze_direction(input_data: dict, output_data: dict) -> None:
    # add gaze estimation (from 3DGazeNet)
    gazes = input_data["narrative_level"]["first_grade_secondary_watcher_looked_system"]["gaze_direction"]
    # cycle every person's eyes positions
    for i in range(len(gazes["eye_pos"])):
        # create the eye_pos dictionary in the format x1, y1, x2, y2 to intend eyes' xy-position
        eye_pos = {}
        eye_pos["x1"] = gazes["eye_pos"][i][0]
        eye_pos["y1"] = gazes["eye_pos"][i][1]
        eye_pos["x2"] = gazes["eye_pos"][i][2]
        eye_pos["y2"] = gazes["eye_pos"][i][3]
        # cycle every "human face object"
        for o in output_data["objects"]:
            if o["type"] != "human face":
                continue
            # create a BoundingBox object based on "human face" position attribute
            face_bbox = BoundingBox(o["position"]["x0"], o["position"]["y0"], o["position"]["x1"], o["position"]["y1"])
            # if the selected bounding box contains every point contained in "eye pos" assign their characteristics to that "human face" object
            if all(i for i in [face_bbox.contains_point(x, y) for x, y in zip(gazes["eye_pos"][i][::2], gazes["eye_pos"][i][1::2])]):
            #if [face_bbox.contains_point(gazes["eye_pos"][i][0], gazes["eye_pos"][i][1]) and face_bbox.contains_point(gazes["eye_pos"][i][2], gazes["eye_pos"][i][3]):
                o["attributes"]["gaze_direction"] = {"eye_pos": eye_pos, "yaw": gazes["yaw"][i], "pitch": gazes["pitch"][i]}
                break


def merge_scene_graph(output_data: dict, input_filename: str, config: dict) -> None:
    # given matching objects between output_data["objects"] and the ones found by an ad hoc model (RelTR), we retrieve the corresponding relations from the latter
    with open(config.filesystem.scene_graph + input_filename + config.filetype.scene_graph, "r") as f:    
        scene_graph = json.load(f)
    # cycle every entry in the scene graph
    for i in scene_graph.keys():
        # create bounding boxes of subject and object
        subj_bbox = BoundingBox(*scene_graph[i]["subject_bbox"])
        obj_bbox = BoundingBox(*scene_graph[i]["object_bbox"])
        # define flags to be checked 
        subject_found = -1
        object_found = -1
        for o in output_data["objects"]:
            # create current object's bounding box
            bbox = BoundingBox(o["position"])
            # if there is a first match between subject's bounding box and the one currently in examination
            if iou(subj_bbox, bbox) > config.threshold.bbox_match and subject_found == -1:
                # save the object's id
                subject_found = o["id"]
            # if there is a first match between object's bounding box and the one currently in examination
            if iou(obj_bbox, bbox) > config.threshold.bbox_match and object_found == -1:
                # save the object's id
                object_found = o["id"]
            # if both subject and object have found a matching bounding box
            if subject_found != -1 and object_found != -1 and subject_found != object_found:
                # add an entry in the "relationships" list and exit the inner loop
                output_data["relationships"].append({
                    "source": subject_found,
                    "target": object_found,
                    "type": scene_graph[i]["relation"]
                })
                break


def add_positional_relations(output_data: dict, config: dict) -> None:
    # add positional relations between objects (source object is above/below/right/left to the target object)
    for i in range(len(output_data["objects"])):
        # skip "human face" objects as they will be managed later
        if output_data["objects"][i]["type"] == "human face":
            continue
        bbox_i = BoundingBox(output_data["objects"][i]["position"]["x0"], output_data["objects"][i]["position"]["y0"], output_data["objects"][i]["position"]["x1"], output_data["objects"][i]["position"]["y1"])
        for j in range(i+1, len(output_data["objects"])):
            # skip "human face" objects as they will be managed later
            if output_data["objects"][j]["type"] == "human face":
                continue
            bbox_j = BoundingBox(output_data["objects"][j]["position"]["x0"], output_data["objects"][j]["position"]["y0"], output_data["objects"][j]["position"]["x1"], output_data["objects"][j]["position"]["y1"])
            relation = ""
            # depth relation between objects: if their bounding boxes are overlapping and a depth value is available
            if iou(bbox_i, bbox_j) > config.threshold.depth_overlap:
                # in front of-position case
                if output_data["objects"][i]["depth"] / output_data["objects"][j]["depth"] > config.threshold.depth_upper_limit:
                    relation = "in front of"
                # behind-position case
                elif output_data["objects"][i]["depth"] / output_data["objects"][j]["depth"] < config.threshold.depth_lower_limit:
                    relation = "behind"
                # every other case of overlapping bounding boxes
                else:
                    relation = "next to"
            else:
                res_y = y_axis_distance(bbox_i, bbox_j)
                if res_y > 0 and abs(res_y) > bbox_j.get_y_dimension() * config.threshold.positional_relation_tolerance:
                    # below-position case
                    relation = "below"
                elif res_y < 0 and abs(res_y) > bbox_j.get_y_dimension() * config.threshold.positional_relation_tolerance:
                    # above-position case
                    relation = "above"
                res_x = x_axis_distance(bbox_i, bbox_j)
                if res_x > 0 and abs(res_x) > bbox_j.get_x_dimension() * config.threshold.positional_relation_tolerance:
                    # right-position case
                    relation = relation+"-right of" if relation != "" else "to the right of"
                elif res_x < 0 and abs(res_x) > bbox_j.get_x_dimension() * config.threshold.positional_relation_tolerance:
                    # left-position case
                    relation = relation+"-left of" if relation != "" else "to the left of"
            if relation != "":
                output_data["relationships"].append({
                    "source": i,
                    "target": j,
                    "type": relation
                })


def add_face_person_relations(output_data: dict, input_filename: str, config: dict) -> None:
    # assignment of "human face" objects to corresponding "person" objects
    # store in a list "person" objects bounding box areas
    persons_bbox = []
    for i in range(len(output_data["objects"])):
        if output_data["objects"][i]["type"] == "person":
            bbox = BoundingBox(output_data["objects"][i]["position"]["x0"], output_data["objects"][i]["position"]["y0"], output_data["objects"][i]["position"]["x1"], output_data["objects"][i]["position"]["y1"])
            persons_bbox.append((i, bbox))
    # sort the bounding boxes by ascending areas
    persons_bbox.sort(key=lambda e: e[1].get_area())
    # semantic segmentation map
    im_frame = Image.open(config.filesystem.panoptic + input_filename + config.filetype.panoptic)
    # panoptic segmentation map
    im_frame_p = Image.open(config.filesystem.panoptic + input_filename + '_panoptic' + config.filetype.panoptic)
    # define numpy arrays to store data maps
    np_frame = np.array(im_frame.getdata()).reshape(im_frame_p.size[1], im_frame_p.size[0])
    # apply a filter that only allows "person" (id = 0) masks to be kept
    np_frame = np.where(np_frame != 0, 0, 1)
    np_frame_p = np.array(im_frame_p.getdata()).reshape(im_frame_p.size[1], im_frame_p.size[0])
    # apply the "person" object mask to the panoptic segmentation map
    seg_map = np_frame_p * np_frame
    for i in range(len(output_data["objects"])):
        # skip non-"human face" objects
        if output_data["objects"][i]["type"] != "human face":
            continue
        bbox_i = BoundingBox(output_data["objects"][i]["position"]["x0"], output_data["objects"][i]["position"]["y0"], output_data["objects"][i]["position"]["x1"], output_data["objects"][i]["position"]["y1"])
        # select which mask contains the "human face" object and keep its id
        instance_val = seg_map[int((bbox_i.get_y1()+bbox_i.get_y0())/2)][int((bbox_i.get_x1()+bbox_i.get_x0())/2)]
        # instance_val will be used as a denominator so it has to be greater than 0
        assert instance_val > 0
        for b in persons_bbox:
            # verify that the "human face" object is contained in the "person" object
            if A_contains_B(b[1], bbox_i):
                # count how many pixels compose the "person" mask in exam
                total_mask_pixels = np.where(seg_map == instance_val, 1, 0).sum()
                # take the bounding box corresponding section of the segmentation map
                bbox_seg_map = seg_map[b[1].get_y0():b[1].get_y1(), b[1].get_x0():b[1].get_x1()]
                # count how many pixels of the "person" mask are contained in the bounding box
                total_bbox_mask_pixels = np.where(bbox_seg_map == instance_val, 1, 0).sum()
                # if the bounding box contains more than a certain percentage of the mask's pixels the "human face" object gets assigned
                if total_bbox_mask_pixels > total_mask_pixels * config.threshold.completely_contained:
                    output_data["relationships"].append({
                        "source": i,
                        "target": b[0],
                        "type": "is part of"
                    })
                    break