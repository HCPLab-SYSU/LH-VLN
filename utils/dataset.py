from torch.utils.data import Dataset, DataLoader
from functools import reduce
import os
import gzip
import json
from PIL import Image
import numpy as np


def read_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return data

class TaskDataset(Dataset):
    def __init__(self, args, mode):
        assert mode in ['train', 'test', 'valid']
        self.mode = mode

        self.task_data = args.task_data
        self.step_task_data = args.step_task_data
         
        if mode == 'train':
            self.batch = args.train_batch
        elif mode == 'valid':
            self.batch = args.val_batch
        else:
            self.batch = args.test_batch

        self.data = []
        for batch in self.batch:
            data = self.load_data(self.task_data, batch)
            for i in range(len(data)):
                subtask = data[i]['Subtask list']
                obj = []
                region_id = []
                for task in subtask:
                    if "Move_to" in task:
                        obj_id  = task[9:-2].split("_")
                        obj.append(obj_id[0])
                        region_id.append(obj_id[1])
                rooms = []
                for room in data[i]["Object"]:
                    rooms.append(room[1].split(': ')[1])
                data[i]['Object'] = obj
                data[i]['Region'] = region_id
                data[i]['Batch'] = batch
                data[i]['Room'] = rooms
            self.data.append(data)

        self.tasks = reduce(lambda x, y: x + y, self.data)
        
        self.step_data = []
        for batch in self.batch:
            self.step_data.append(self.load_step_task(batch))
        self.step_tasks = reduce(lambda x, y: x + y, self.step_data)
    
    def __getitem__(self, index):
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return self.tasks[index], self.step_tasks[index]

    
    def __len__(self):
        return len(self.tasks)
    
    def load_data(self, file, batch):
        task = []
        file = file + batch
        nums = os.listdir(file)
        for num in nums:
            task_names = os.listdir(file + "/" + num)
            for task_name in task_names:
                f = file + "/" + num + "/" + task_name + "/config.json"
                with open(f, "r", encoding='utf-8') as r:
                    task.append(json.load(r))
        return task 

    def load_step_task(self, batch):
        step_tasks = []
        path = self.step_task_data + batch
        index = self.batch.index(batch)
        for task in self.data[index]:
            step_task_list = os.listdir(path)
            step_tasks_list = []
            for step_task in step_task_list:
                step_task_path = path + "/" + step_task
                with open(step_task_path, "r", encoding='utf-8') as r:
                    config = json.load(r)
                if task['Task instruction'] in config["trajectory path"]:
                    config["trajectory path"] = self.task_data + batch + '/' + str(len(task['Object'])) + '/' + '/'.join(config["trajectory path"].split('/')[-3:])
                    config['Batch'] = batch
                    config['Object'] = config['target']

                    step_tasks_list.append(config)
            step_tasks.append(step_tasks_list)
        
        return step_tasks

      
    def preprocess(self, data):
        # 将data 做一些预处理
        pass

class EpisodeDataset(Dataset):
    def __init__(self, args, mode):
        assert mode in ['train', 'test', 'valid']
        self.mode = mode
        self.episode_data = args.episode_data

        if mode == 'train':
            self.batch = args.train_batch
        elif mode == 'valid':
            self.batch = args.val_batch
        else:
            self.batch = args.test_batch

        self.data = []
        self.step_data = []
        for batch in self.batch:
            data, step_data = self.load_data(self.episode_data, batch)
            for i in range(len(data)):
                subtask = data[i]['Subtask list']
                obj = []
                region_id = []
                for task in subtask:
                    if "Move_to" in task:
                        obj_id  = task[9:-2].split("_")
                        obj.append(obj_id[0])
                        region_id.append(obj_id[1])
                rooms = []
                for room in data[i]["Object"]:
                    rooms.append(room[1].split(': ')[1])
                data[i]['Object'] = obj
                data[i]['Region'] = region_id
                data[i]['Batch'] = batch
                data[i]['Room'] = rooms
            self.data.append(data)
            self.step_data.append(step_data)

        self.tasks = reduce(lambda x, y: x + y, self.data)
        self.step_tasks = reduce(lambda x, y: x + y, self.step_data)
    
    def __getitem__(self, index):
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return self.tasks[index], self.step_tasks[index]

    
    def __len__(self):
        return len(self.tasks)
    
    def load_data(self, file, batch):
        task = []
        step_tasks = []
        file = file + batch + '.json.gz'
        raw_dic = read_json_gz(file)
        for key, value in raw_dic.items():
            task.append(value['lh_task'])
            step_tasks_list = []
            for step_task in value['st_task']:
                step_task['Batch'] = batch
                step_task['Object'] = step_task['target']
                step_tasks_list.append(step_task)
            step_tasks.append(step_tasks_list)
        return task, step_tasks

    def preprocess(self, data):
        # 将data 做一些预处理
        pass

class SFTDataset(Dataset):
    def __init__(self, args, mode, max_len=None):
        assert mode in ['train', 'test', 'valid']
        self.mode = mode
        
        self.task_data = args.task_data
        self.step_task_data = args.step_task_data

        if mode == 'train':
            self.batch = args.train_batch
        elif mode == 'valid':
            self.batch = args.val_batch
        else:
            self.batch = args.test_batch

        self.data = []
        for batch in self.batch:
            data = self.load_data(self.task_data, batch)
            self.data.append(data)

        self.tasks = reduce(lambda x, y: x + y, self.data)
        
        self.step_data = []
        for batch in self.batch:
            self.step_data.append(self.load_step_task(batch))
        self.step_tasks = reduce(lambda x, y: x + y, self.step_data)
    
    def __getitem__(self, index):
        return self.preprocess(self.tasks[index], self.step_tasks[index])

    
    def __len__(self):
        return len(self.tasks)
    
    def load_data(self, file, batch):
        task = []
        file = file + batch
        nums = os.listdir(file)
        for num in nums:
            task_names = os.listdir(file + "/" + num)
            for task_name in task_names:
                f = file + "/" + num + "/" + task_name + "/success/trial_1/task.json"
                with open(f, "r", encoding='utf-8') as r:
                    t = json.load(r)
                t["trajectory path"] = file + "/" + num + "/" + task_name + "/success/trial_1"
                task.append(t)
        return task 

    def load_step_task(self, batch):
        step_tasks = []
        path = self.step_task_data + batch
        index = self.batch.index(batch)
        for task in self.data[index]:
            step_task_list = os.listdir(path)
            step_tasks_list = []
            for step_task in step_task_list:
                step_task_path = path + "/" + step_task
                with open(step_task_path, "r", encoding='utf-8') as r:
                    config = json.load(r)
                if task['Task instruction'] in config["trajectory path"]:
                    config["trajectory path"] = self.task_data + batch + '/' + str(len(task['Object'])) + '/' + '/'.join(config["trajectory path"].split('/')[-3:])
                    config['Batch'] = batch
                    config['Object'] = config['target']

                    step_tasks_list.append(config)
            step_tasks.append(step_tasks_list)
        
        return step_tasks
      
    def preprocess(self, task, step_tasks):
        task_trajectory = task['trajectory path']
        assert task_trajectory == step_tasks[0]['trajectory path']

        action_list = []
        for action in os.listdir(task_trajectory):
            if 'json' in action:
                continue
            action_list.append(action)
        action_list = sort_step_action_list(action_list)

        pos_list = []
        rot_list = []
        trial_keys = sorted(task["trial"].keys(), key=lambda x: int(x.split('_')[1]))
    
        for trial_key in trial_keys:
            trial_data = task["trial"][trial_key]
            pos_list.extend(trial_data["pos"])
            rot_list.extend(trial_data["yaw"])

        if len(pos_list) != len(action_list):
            return None, None
        
        task_traj = []
        for i in range(len(action_list)-1):
            action_label = "_".join(action_list[i+1].split('_')[1:-2])
            pos = pos_list[i]
            rot = rot_list[i]
            obs = []
            for direction in ['left', 'front', 'right']:
                img_path = os.path.join(task_trajectory, action_list[i], f'{direction}.png')
                if os.path.exists(img_path):
                    obs.append(Image.open(img_path))
                else:
                    obs.append(None)

            task_traj.append({
                'action': action_label,
                'position': pos,
                'rotation': rot,
                'observation': obs
            })
        
        proccessed_task = {
            'Task instruction': task['Task instruction'],
            'Object': task['Object'],
            'Room': task['Region Name'],
            'Trajectory': task_traj,
        }

        proccessed_step_tasks = []
        for step_task in step_tasks:
            start, end = step_task['start'], step_task['end']
            index_start = None
            index_end = None
            for i, action in enumerate(action_list):
                step = int(action.split('_')[0])
                if step == start and index_start is None:
                    index_start = i
                if step == end:
                    index_end = i
            
            if index_start is not None and index_end is not None:
                proccessed_step_tasks.append({
                    'Task instruction': step_task['Task instruction'],
                    'Object': step_task['Object'],
                    'Room': [task['Region Name'][task['Region'].index(step_task['Region'][0])]],
                    'Trajectory': task_traj[index_start:index_end+1],
                })
        
        return proccessed_task, proccessed_step_tasks

def sort_step_action_list(lst):
    def sort_key(item):
        parts = item.split('_')
        step = int(parts[0])  
        action = parts[1]     
        
        return (step, action == "stop")
    
    return sorted(lst, key=sort_key)
    
def split_datasets_by_scene(dataset_list):
    """
    Split multiple TaskDataset or EpisodeDataset instances into three datasets 
    based on 'Scene' labels in self.tasks elements
    
    Args:
        dataset_list: List containing multiple TaskDataset or EpisodeDataset instances
    
    Returns:
        tuple: (dataset_0_700, dataset_700_800, dataset_800_plus) Three redistributed dataset lists
    """
    # Collect all tasks and step_tasks from all datasets
    all_tasks = []
    all_step_tasks = []
    
    for dataset in dataset_list:
        all_tasks.extend(dataset.tasks)
        all_step_tasks.extend(dataset.step_tasks)
    
    # Classify by Scene labels
    tasks_0_700 = []
    step_tasks_0_700 = []
    tasks_700_800 = []
    step_tasks_700_800 = []
    tasks_800_plus = []
    step_tasks_800_plus = []
    
    for i, task in enumerate(all_tasks):
        # Extract Scene ID from task, assume it's an integer field
        scene_id = int(task['Scene'][:5])  

        if 0 <= scene_id < 700:
            tasks_0_700.append(task)
            step_tasks_0_700.append(all_step_tasks[i])
        elif 700 <= scene_id < 800:
            tasks_700_800.append(task)
            step_tasks_700_800.append(all_step_tasks[i])
        else:  # scene_id >= 800
            tasks_800_plus.append(task)
            step_tasks_800_plus.append(all_step_tasks[i])
    
    return (tasks_0_700, step_tasks_0_700), (tasks_700_800, step_tasks_700_800), (tasks_800_plus, step_tasks_800_plus)

def create_split_datasets(dataset_list, args):
    """
    Create three new dataset instances containing data split by Scene
    
    Args:
        dataset_list: Original dataset list
        args: Original arguments
    
    Returns:
        tuple: Three new dataset instances
    """
    # Get split data
    (tasks_0_700, step_tasks_0_700), (tasks_700_800, step_tasks_700_800), (tasks_800_plus, step_tasks_800_plus) = split_datasets_by_scene(dataset_list)
    
    # Determine dataset class type
    dataset_class = type(dataset_list[0])
    
    # Create three new dataset instances
    dataset_0_700 = dataset_class.__new__(dataset_class)
    dataset_700_800 = dataset_class.__new__(dataset_class)
    dataset_800_plus = dataset_class.__new__(dataset_class)
    
    mode = ['train', 'valid', 'test']
    # Initialize basic attributes for all three datasets
    for i, dataset in enumerate([dataset_0_700, dataset_700_800, dataset_800_plus]):
        dataset.mode = mode[i]
        
        # Copy attributes based on dataset type
        if hasattr(dataset_list[0], 'task_data'):
            dataset.task_data = dataset_list[0].task_data
        if hasattr(dataset_list[0], 'step_task_data'):
            dataset.step_task_data = dataset_list[0].step_task_data
        if hasattr(dataset_list[0], 'episode_data'):
            dataset.episode_data = dataset_list[0].episode_data
        
        # Set batch size based on mode
        if mode == 'train':
            dataset.batch = args.train_batch
        elif mode == 'valid':
            dataset.batch = args.val_batch
        else:
            dataset.batch = args.test_batch
    
    # Assign task data to each dataset
    dataset_0_700.tasks = tasks_0_700
    dataset_0_700.step_tasks = step_tasks_0_700
    dataset_0_700.data = []  # Can be reorganized based on needs
    dataset_0_700.step_data = []
    
    dataset_700_800.tasks = tasks_700_800
    dataset_700_800.step_tasks = step_tasks_700_800
    dataset_700_800.data = []
    dataset_700_800.step_data = []
    
    dataset_800_plus.tasks = tasks_800_plus
    dataset_800_plus.step_tasks = step_tasks_800_plus
    dataset_800_plus.data = []
    dataset_800_plus.step_data = []
    
    return dataset_0_700, dataset_700_800, dataset_800_plus
