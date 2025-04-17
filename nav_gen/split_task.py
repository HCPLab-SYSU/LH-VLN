import numpy as np
import random
import os 
import sys
import torch
import json
import logging
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from recognize_anything.ram.models import ram_plus
from recognize_anything.ram import inference_ram as inference
from recognize_anything.ram import get_transform
from gpt import gpt3_5, gpt4
from tqdm import tqdm


def init_model(args):
    global transform 
    global ram_model 
    global device

    pretrained = args.ram_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = get_transform(image_size=384)

    ram_model = ram_plus(pretrained=pretrained,
                                image_size=384,
                                vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.ram_logs)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)

    # redirect stdout and stderr to logging module
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

class TrainDataset(Dataset):
    def __init__(self, task_data, max_len=None):
        self.data_path = task_data
        self.max_len = max_len

        self.task_list = self.make_task_list()

    def make_task_list(self):
        task_list = []
        task_type = os.listdir(self.data_path)
        for type_ in task_type:
            type_task_list = os.listdir(self.data_path + type_)
            for type_task in type_task_list:
                task_list.append(type_ + '/' + type_task)
        
        if self.max_len:
            task_list = task_list[:self.max_len]

        return task_list

    def make_task_dic(self):
        task_dic = {}
        task_type = os.listdir(self.data_path)
        for type_ in task_type:
            task_dic[type_] = os.listdir(self.data_path + type_)

        return task_dic
    
    def __getitem__(self, index):
        # return the path of the task
        task_path = self.data_path + self.task_list[index]
        return task_path


def get_trail(path):
    action_path = path + '/success/trial_1'
    actions = os.listdir(action_path)
    action_dic = {}
    for action in actions:
        if 'task' not in action:
            action_dic[int(action.split('_')[0])] = ["_".join(action.split('_')[1:-2]), action.split('_')[-1]]  
    obs_dic = {}

    for key, value in action_dic.items():
        if key == -1:
            continue

        img_list = os.listdir(action_path + '/' + str(key-1) + "_" + action_dic[key-1][0] + "_for_" + action_dic[key-1][1])
        obs = {}
        for img in img_list:
            img_name = img.split('.')[0]
            img_path = action_path + '/' + str(key-1) + "_" + action_dic[key-1][0] + "_for_" + action_dic[key-1][1] + '/' + img
            obs[img_name] = img_path

        obs_dic[key] = obs

    action_ = action_dic[key][0]
    obj_ = action_dic[key][1]

    removed_value = action_dic.pop(-1, None)

    sorted_action_dic = sorted(action_dic.items(), key=lambda item: item[0])
    sorted_obs_dic = sorted(obs_dic.items(), key=lambda item: item[0])

    return dict(sorted_action_dic), dict(sorted_obs_dic)

def batch_ram(img_list):
    tag_dic = {}
    for img in img_list:
        image = transform(Image.open(img)).unsqueeze(0).to(device)
        res = inference(image, ram_model)
        tags = res[0].split(' | ')
        for tag in tags:
            if any(sub in tag for sub in ['ceiling', 'floor']):
                continue
            if tag not in tag_dic:
                tag_dic[tag] = 1
            else:
                tag_dic[tag] += 1

    top_five_tags = [key for key, value in sorted(tag_dic.items(), key=lambda item: item[1], reverse=True)[:5]]

    return top_five_tags

def segment_trajectory(trajectory, obs_dic, task, key):
    target = trajectory[0][1]
    trajectory = [tjc[0] for tjc in trajectory]
    n = len(trajectory)
    segments = []
    
    # Step 1: Identify same direction segments
    i = 0
    while i <= n - 3:
        window = trajectory[i:i + 3]
        if window.count('turn_left') >= 2:
            left_indices = [idx for idx, action in enumerate(window) if action == 'turn_left']
            start = i + left_indices[0]
            end = i + left_indices[-1]
            segments.append((start, end, 'turn_left'))
            i = end + 1  # Move past this segment
        else:
            i += 1
    i = 0
    while i <= n - 3:
        window = trajectory[i:i + 3]
        if window.count('turn_right') >= 2:
            right_indices = [idx for idx, action in enumerate(window) if action == 'turn_right']
            start = i + right_indices[0]
            end = i + right_indices[-1]
            segments.append((start, end, 'turn_right'))
            i = end + 1  # Move past this segment
        else:
            i += 1
            
    # Step 2: Merge overlapping or adjacent same direction segments
    merged_segments = []
    if segments:
        segments.sort()
        current_start, current_end, current_label = segments[0]
        for segment in segments[1:]:
            start, end, label = segment
            if start <= current_end + 3 and label == current_label:
                # merge segments
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end, current_label))
                current_start, current_end, current_label = segment
                
        merged_segments.append((current_start, current_end, current_label))
                
    # Step 3: Handle overlapping different direction segments
    result_segments = []
    temp = merged_segments[:]
            
    while len(temp) > 1:
        curr_start, curr_end, curr_label = temp.pop(0)
        next_start, next_end, next_label = temp.pop(0)
        if curr_end + 1 >= next_start:
            # overlapping segments
            new_start = min(curr_start, next_start)
            new_end = max(curr_end, next_end)
            result_segments.append((new_start, new_end, "zigzag"))
        else:
            result_segments.append((curr_start, curr_end, curr_label))
            temp.insert(0, (next_start, next_end, next_label))
    if temp:
        result_segments += temp

    # Step 4: Mark `move_forward` segment 
    final_segments = []
    last_end = -1    
    for seg_start, seg_end, label in result_segments:
        if last_end + 2 < seg_start:
            final_segments.append({
                "trajectory": task,
                "start": last_end + 1 + key,
                "end": seg_start - 2 + key,
                "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, seg_start - 1 + key)],
                "label": "move_forward",
                "target": target   
                })
        final_segments.append({
            "trajectory": task,
            "start": seg_start - 1 + key,
            "end": seg_end + key,
            "obs": [obs_dic[i]['front'] for i in range(seg_start - 1 + key, seg_end + key + 1)],
            "label": label if trajectory[seg_start - 1 : seg_end + 1].count(label) <=3 else "make a " + label.split("_")[-1] + " turn",
            "target": target
                })
        last_end = seg_end

    # Append last `move_forward` segment if needed
    if last_end < n - 1:
        final_segments.append({
            "trajectory": task,
            "start": last_end + 1 + key,
            "end": n - 1 + key,
            "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, n + key)],
            "label": "move_forward",
            "target": target
            })

    for seg in final_segments:
        seg['scene tags'] = batch_ram(seg['obs'])
        del seg['obs']

    return final_segments

def make_task(args, trail_list):
    for trail in tqdm(trail_list, desc="Make Task"):
        tags = {}
        tags['target'] = trail[0]['target']
        for i in range(len(trail)):
            tags['step_' + str(i)] = {
                'action': trail[i]['label'],
                'tags': trail[i]['scene tags']
            }
        task = {}
        with open(trail[0]['trajectory'] + "/success/trial_1/task.json", "r", encoding='utf-8') as r:
            config = json.load(r)
        task["trajectory path"] = '/'.join(trail[0]['trajectory'].split('/')[-3:]) + '/success/trial_1'
        task["start"] = trail[0]['start']
        task["end"] = trail[-1]['end']
        task["Robot"] = config['Robot']
        task["Scene"] = config['Scene']
        task["target"] = trail[0]['target']
        index = config["Object"].index(task["target"])
        task["target"] = [config["Object"][index]]
        task['Region'] = [config["Region"][index]]
        step_former = 0
        for i in range(0, index):
            trail_name = 'trial_' + str(i)
            step_former += len(config["trial"][trail_name]['pos'])
        task['start_pos'] = config["trial"]['trial_' + str(index)]['pos'][task["start"]-step_former]
        task['start_yaw'] = config["trial"]['trial_' + str(index)]['yaw'][task["start"]-step_former]
        task["Task instruction"] = gpt4(args, args.prompt_path + "gen_task.txt", json.dumps(tags))
        print(task)

        task_root_path = args.step_task_path
        if len(task["Task instruction"]) > 100:
            ins_path = task["Task instruction"][:100]
        else:
            ins_path = task["Task instruction"]
        task_path = task_root_path + ins_path + '.json' if ins_path[-1] != ' ' else task_root_path + ins_path[:-1] + '.json'
        with open(task_path, 'w') as json_file:
            json.dump(task, json_file, indent=4)

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)
         

def split_traj(args):
    init_model(args)
    task_data = args.task_path
    task_dataset = TrainDataset(task_data)

    trail_list = []
    for task in tqdm(task_dataset, desc="Segment Trajectory"):
        print(task)
        action_dic, obs_dic = get_trail(task)
        # print(obs_dic)
        # print(action_dic)
        start = 0
        end = start + 1
        target = action_dic[start][1]
        while end < len(obs_dic):
            # 完成一个轨迹
            if action_dic[end][1] != target or end == len(obs_dic) - 1:
                if end - start > 5:
                    trail = [action_dic[i] for i in range(start, end)]
                    patch_trail = segment_trajectory(trail, obs_dic, task, start)
                    trail_list.append(patch_trail)
                start = end
                end += 1
                target = action_dic[start][1]
            if end - start == 1:
                if  start < len(obs_dic) - 3:
                    if [action_dic[i][0] for i in range(start, start+3)].count('move_forward') == 3:
                        start -= 1
                start += 1

            end += 1  
              
    with open(args.split_save_path, 'w') as file:
        for item in trail_list:
            file.write(f'{item}\n')

def gen_step_task(args):
    init_model(args)
    import ast
    with open(args.split_save_path, 'r') as file:
        trail_list = [ast.literal_eval(line.strip()) for line in file]
        
    # print(trail_list)
    make_task(args, trail_list)
