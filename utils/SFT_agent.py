from .common_utils import transform_position, transform_rotation, transback_position, transback_rotation, rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import numpy as np
from EVA.EVA_CLIP.rei.clip import get_image_embedding
import math
import torch

label_index = {
    "stop": np.array([0]), # stop
    "turn_left": np.array([1]),
    "move_forward": np.array([2]),
    "turn_right": np.array([3]),
}


# it contains the training and validation loop
class SFTAgent:
    def __init__(self, args, config, nav_model):
        self.args = args
        self.config = config
        self.nav_model = nav_model
        
    def train(self, criterion):
        # init
        ml_loss = 0.
        count = 0
        action = 'stop'
        ins = self.config["Task instruction"]

        for i, step in enumerate(self.config["Trajectory"]):
            agent_pos = transform_position(step["position"])
            agent_rot = euler_angles_to_rotation_matrix(np.array([0, 0, math.degrees(step["rotation"] - 180)]))
            obs = step["observation"]

            input = {
                'observations': [
                    {
                        "instruction": self.task_sim.ins,                                       # instruction
                        "view_feats": get_image_embedding(obs),                                 # "view features"
                        "pose": np.append(agent_pos, rotation_matrix_to_euler_angles(agent_rot)[2]),    # xyz,heading
                    }
                ],
            }

            label = step["action"]
            label_onehot = torch.from_numpy(label_index[label])
            # print(input)
            action, output = self.nav_model.step(input["observations"])
            # output = torch.from_numpy(label_index[action])

            # print gt action
            print(f"The ground truth action for step {i} is {label}")
                    
            cnt_loss = criterion(output, label_onehot.to(output.device)) / self.args.gradient_accumulation_step

            ml_loss += cnt_loss.detach()
            count += 1

            cnt_loss.backward() 
            
        return ml_loss/count if count > 0 else ml_loss