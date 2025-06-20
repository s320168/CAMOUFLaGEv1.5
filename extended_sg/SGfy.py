import json
from os import listdir
from os.path import isfile, join
from utils.parser import Parser
from utils.utils import add_objects, add_persons_characteristics, add_faces_emotions, add_head_pose, add_gaze_direction, add_positional_relations, add_face_person_relations, add_objects_depth, merge_scene_graph
from tqdm import tqdm

def list_objects(input_data: dict, output_data: dict, input_filename: str, config: dict) -> None:
    # add the picture's environment to the graph (could it be useful?)
    output_data["scene"] = {
        # brief description of the scene captured in the image
        "single_action_caption": input_data["figurative_level"]["action"]["single_action_caption"],
        # specifies if the image depicts an indoor or outdoor environment
        "indoor-outdoor": input_data["narrative_level"]["basic_watcher_looked_system"]["scene"]["indoor-outdoor"],
        # color palette identified in the image
        "color_palette": input_data["plastic_level"]["chromatic_categories"]["colors"]["palette"],
        # image's dimension expressed in width and height
        "dimensions": input_data["technical"]["dimensions"]
    }
    
    # if no objects are detected something's wrong and the graph isn't possible
    assert len(input_data["plastic_level"]["topological_categories"]["obj_positions"]) > 0
    output_data["objects"] = []
    # adds the object entries to the "objects" list
    add_objects(input_data, output_data)
    
    # adds depth attribute to the objects previously identified
    add_objects_depth(input_data, output_data, input_filename, config)
    
    # adds gender, age and ethnicity to "human face" objects
    add_persons_characteristics(input_data, output_data, config)
    
    # adds to "human face" objects their dominant emotion
    add_faces_emotions(input_data, output_data, config)
    
    # adds to "human face" objects their head pose expressed in yaw, pitch and roll
    add_head_pose(input_data, output_data, config)

    # adds to "human face" objects their gaze direction expressed in yaw and pitch
    add_gaze_direction(input_data, output_data)
        

def list_relations(output_data: dict, input_filename: str, config: dict) -> None:
    # if less than two objects are detected something's wrong and relations aren't possible
    #assert len(output_data["objects"]) >= config.threshold.min_objects_for_relations
    output_data["relationships"] = []

    # add relations found by an ad hoc model (RelTR)
    merge_scene_graph(output_data, input_filename, config)

    # computes positional relations like being in front of, behind, to the left, to the right, above, below, ...
    add_positional_relations(output_data, config)

    # computes to which "person" a "human face" object belongs
    add_face_person_relations(output_data, input_filename, config)
    

if __name__ == "__main__":
    # default config path
    config_path = "config/config.yaml"   
    parser = Parser(config_path)
    # read configuration file and parse possible arguments
    config = parser.parse_args()
    # construct a list containing filenames about the input directory while skipping other directories
    input_files = [f.replace(config.filetype.input, "").replace("identikit-", "") for f in listdir(config.filesystem.input) if isfile(join(config.filesystem.input, f))]
    # cycle every input file
    for file in tqdm(input_files):
        # load the identikit input file into a local dictionary
        with open(join(config.filesystem.input, "identikit-" + file + config.filetype.input), "r") as f:
            identikit = json.load(f)
        # define the output dictionary
        extended_sg = dict()
        # call the function that will create the object list
        list_objects(identikit, extended_sg, file, config)
        # call the function that will create the objects' relations list
        list_relations(extended_sg, file, config)
        # save the final result in a JSON file
        output = json.dumps(extended_sg, indent=True)
        with open(join(config.filesystem.output, "extended_sg_" + file + config.filetype.output), "w+") as f:
            f.write(output)