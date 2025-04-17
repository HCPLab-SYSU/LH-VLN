import numpy as np
import itertools
import math
import random
import json
import copy
import csv
import ast
import os

from gpt import gpt4

def sample_obj(scene_dic):
    for key, value in scene_dic.items():
        if len(value) <= 5:
            continue
        else:
            scene_dic[key] = random.sample(value, 5)

    return scene_dic


def get_mid_distance_nodes(node_dict):
    """
    Select three nodes from the dictionary whose pairwise distances are in the middle range.

    Parameters:
        node_dict (dict): A dictionary where keys are node names and values are 3D coordinates.

    Returns:
        list: Names of three nodes with mid-range distances.
    """

    def calculate_distance(coord1, coord2):
        """Calculate the Euclidean distance between two 3D points."""
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))

    # Get all node names and their coordinates
    nodes = list(node_dict.keys())
    coordinates = list(node_dict.values())

    # Compute distances between all pairs of nodes
    distances = []
    for (i, coord1), (j, coord2) in itertools.combinations(enumerate(coordinates), 2):
        distance = calculate_distance(coord1, coord2)
        distances.append((distance, nodes[i], nodes[j]))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])

    # Find the pair of nodes with a mid-range distance
    mid_index = len(distances) // 2
    mid_distance_pair = distances[mid_index]

    # Select the two nodes from the mid-range distance pair
    selected_nodes = {mid_distance_pair[1], mid_distance_pair[2]}

    # Find a third node that has a distance close to the mid-range distance
    remaining_nodes = set(nodes) - selected_nodes
    third_node = None
    min_diff = float('inf')

    for node in remaining_nodes:
        node_coord = node_dict[node]
        # Calculate distances from the current node to the selected nodes
        dist_to_selected = [
            calculate_distance(node_dict[selected_node], node_coord)
            for selected_node in selected_nodes
        ]
        # Compute the average distance to the selected nodes
        avg_distance = sum(dist_to_selected) / len(dist_to_selected)
        # Find the node with the smallest difference from the mid-range distance
        diff = abs(avg_distance - mid_distance_pair[0])
        if diff < min_diff:
            min_diff = diff
            third_node = node

    # Add the third node to the selected nodes
    selected_nodes.add(third_node)
    return list(selected_nodes)


def sample_region(scene_dic, sample_dic):
    if len(scene_dic) <= 3:
        return scene_dic
    else:
        # randomly select three regions
        sample_keys = get_mid_distance_nodes(sample_dic)
        # create a new dictionary with the selected regions
        new_dict = {key: scene_dic[key] for key in sample_keys}
        return new_dict


def read_scene(file_path):
    scene_region = {}
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            if row[-1] != " Unknown room":
                region = "Region " + row[1] + ": " + row[-1][1:]
                if row[0] not in scene_region:
                    scene_region[row[0]] = [region]
                else:
                    scene_region[row[0]].append(region)

    return scene_region


# csv_file = './scene/Per_Scene_Total_Weighted_Votes.csv'


def gen_task(args):
    useless = ["wood", "cleaner", "door knob", "mirror", "switch", "device"]
    scene_region = read_scene(args.region_file)

    if args.scene_id:
        sample_scene = args.scene_id
    else:
        sample_scene = random.sample(scene_region.keys(), 1)
    robot_list = ['spot', 'stretch']
    robot = random.sample(robot_list, 1)

    with open(args.scene_path + sample_scene[0] + ".txt", "r") as f:
        input_scene = {}
        for region in scene_region[sample_scene[0]]:
            input_scene[region] = []

        scene_position = copy.deepcopy(input_scene)

        key = list(input_scene.keys())[0]
        for line in f.readlines():
            line = line.strip('\n')  # strip the newline character
            if "Region" in line:
                for k in list(input_scene.keys()):
                    num = line.split('_')[-1].split(',')[0]
                    if num in k:
                        key = k
                        break
            if "name" in line:
                line_split = line.split(",")
                name = line_split[1][6:]
                if name in input_scene[key] or name in useless:
                    continue
                else:
                    input_scene[key].append(name.replace('/', ''))
                    scene_position[key].append(ast.literal_eval(line.split("position:")[-1]))

    if args.sample_region:
        for key, value in scene_position.items():
            scene_position[key] = np.mean(np.array(value), axis=0)

        input_scene = sample_region(input_scene, scene_position)

    if args.sample_obj:
        input_scene = sample_obj(input_scene)

    with open(args.prompt_path + robot[0] + ".txt", "r") as f:
        input_robot = f.read()

    with open(args.prompt_path + "rule.txt", "r") as f:
        prompt_rule = f.read()

    with open(args.prompt_path + "example.txt", "r") as f:
        prompt_example = f.read()

    prompt_input = "Please observe the above rules strictly. Think step by step.\nINPUT:\n```\nscene: " + json.dumps(input_scene) + "\nrobot: " + json.dumps(input_robot) + "\n```\nOUTPUT:"
    
    prompt = prompt_rule + "\n" + prompt_example + "\n" + prompt_input
    print(prompt)
    
    task = gpt4(args, args.prompt_path + "system.txt", prompt)
    if '```' in task:
        task = task[3:-3]
    if "python" in task:
        task = task[6:]
    print(task)
    task_dic = json.loads(task)
    task_dic["Robot"] = robot[0]
    task_dic["Scene"] = sample_scene[0]
    print(task_dic)
    wrong = False
    objs = []
    for task in task_dic["Subtask list"]:
        if "Move_to" in task:
            obj_id = task[9:-2]
            obj = obj_id.split("_")
            ok = None
            for k in list(input_scene.keys()):
                if obj[1] in k:
                    ok = k
                    break
            if ok == None:
                wrong = True
                break
            objs.append([obj[0], ok])
            if obj[0] not in input_scene[ok]:
                wrong = True
                break
    if "region" in task_dic['Task instruction'] or "Region" in task_dic['Task instruction']:
        wrong = True
    if wrong:
        return False
    length = len(objs)
    task_dic['Object'] = objs
    if task_dic['Task instruction'].endswith('.') or task_dic['Task instruction'].endswith('?'):
        task_dic['Task instruction'] = task_dic['Task instruction'][:- 1]
    file = task_dic['Task instruction']
    if not os.path.isdir(args.task_path + str(length)):
        os.mkdir(args.task_path + str(length))
    if not os.path.isdir(args.task_path + str(length) + '/' + file):
        os.mkdir(args.task_path + str(length) + '/' + file)
    with open(args.task_path + str(length) + '/' + file + '/config.json', 'w') as json_file:
        json.dump(task_dic, json_file, indent=4)
    print("saved")
    
    return task_dic