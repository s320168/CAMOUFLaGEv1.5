## Extended Scene Graph Adapter

This module takes in input FRESCO and RelTR's output to create an extended scene graph, which will be then used to condition Stable Diffusion generation process.

### Input

#### Identikit

JSON file containing various analysis results about an image: objects detected, scene caption, color information, depth estimation an many more.

At the moment the data used is contained in the following fields:
- `plastic_level.topological_categories.obj_positions`;
- `plastic_level.chromatic_categories.colors.palette`;
- `figurative_level.action.single_action_caption`;
- `figurative_level.content_participants.single_person_characteristics`;
- `figurative_level.content_participants.single_person_face_attributes`;
- `figurative_level.emotion.emotion.emotion_scores`;
- `narrative_level.first_grade_secondary_watcher_looked_system`;
- `narrative_level.basic_watcher_looked_system.scene.place`;
- `narrative_level.first_grade_secondary_watcher_looked_system.head_pose`;
- `narrative_level.first_grade_secondary_watcher_looked_system.gaze_direction`.

#### Maps

PNG files containing info about panoptic semgentation and depth estimation.

#### Scene Graph

JSON file containing subject, relation, object, subject bounding box and object bounding box to add relations to the extended scene graph.

### Output

The final output is a JSON file structured as described:

#### Scene

- `single_action_caption`: a brief caption about the image;
- `indoor-outdoor`: labels the scene as depicting an indoor/outdoor environment;
- `color_palette`: a list of the most prevalent colors in the image;
- `dimensions`: width and height of the image

#### Objects

Each object is characterized by:

- `id`: unique identifier;
- `type`: label of the detected object;
- `position`: contains the upper-left and lower-right points of the object's bounding box;
- `depth`: the average depth of the object.

`human face`-type objects contain an extension list of attributes composed of:

- `face_attributes`: a list of traits that describe the face;
- `attributes`: 
    - `age`: estimates how old the subject is;
    - `gender_scores`: a list of the subject's gender scores;
    - `ethnicity_scores`: a list of the subject's ethnicity scores;
    - `emotion_scores`: a list of the subject's emotion scores;
    - `head_pose`: describes the corresponding head position by evaluating `yaw`, `pitch` and `roll`;
    - `gaze_direction`: describes the corresponding gaze direction by evaluating `eye_pos`, `yaw` and `pitch`;

Recognizable objects:

- [Prismer Object Detection](https://gitlab.com/grains2/fresco/-/tree/main/bin/prismer/prismer?ref_type=heads)

#### Relationships

Describe which relation connects two objects by defining:

- `source`: id of the relation's subject;
- `target`: id of the relation's object;
- `type`: labels the relation;

Recognizable relations:

- `in front of`;
- `behind`;
- `next to`;
- `in front of`;
- `below`;
- `above`;
- `to the right of`;
- `to the left of`;
- `below-right of`;
- `below-left of`;
- `above-right of`;
- `above-left of`;
- [Visual Genome relations](https://github.com/yrcong/RelTR/blob/fca7397e9aaeccd95541e83afa4b971f3fa89014/inference.py#L112C5-L112C16);