from torch.utils.data import Dataset
from functools import reduce
import os
import json
import habitat_sim
import numpy as np


class TaskDataset(Dataset):
    def __init__(self, args):
        self.task_data = args.task_path
        self.step_task_data = args.step_task_path

        self.data = []

        data = self.load_data(self.task_data)
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
            data[i]['Room'] = rooms
        self.data.append(data)

        self.tasks = reduce(lambda x, y: x + y, self.data)

    
    def __getitem__(self, index):
        # data = self.preprocess(self.data[index]) # if you need to preprocess the data
        return self.tasks[index]
    
    def __len__(self):
        return len(self.tasks)
    
    def load_data(self, file):
        task = []
        nums = os.listdir(file)
        for num in nums:
            task_names = os.listdir(file + num)
            for task_name in task_names:
                f = file + num + "/" + task_name + "/config.json"
                with open(f, "r", encoding='utf-8') as r:
                    task.append(json.load(r))
        return task 

    def preprocess(self, data):
        # 将data 做一些预处理
        pass


def geodesic_distance(pathfinder, pos_a, pos_b):
    path = habitat_sim.nav.ShortestPath()
    path.requested_end = np.array(
        np.array(pos_b, dtype=np.float32)
    )

    path.requested_start = np.array(pos_a, dtype=np.float32)

    if pathfinder.find_path(path):
        geo_dis = path.geodesic_distance
        return geo_dis
    else:
        return None

def count_gt_length(args, sim):
    dis = []
    data_path = args.task_path

    length = len(sim.target)
    task_path = data_path + str(length) + '/' + sim.ins + '/success/trial_1/task.json'

    with open(task_path) as f:
      task_config = json.load(f)
    
    for i in range(length):
        traj =task_config["trial"]["trial_"+str(i)]
        dis.append(geodesic_distance(sim.pathfinder, traj["pos"][0], traj["pos"][-1]))
    return dis