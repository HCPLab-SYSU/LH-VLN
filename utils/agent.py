from habitat_base.simulation import SceneSimulator
from .common_utils import transform_position, transform_rotation, transback_position, transback_rotation, rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import numpy as np
import json
import math
import shutil
import torch
import os

label_index = {
    "stop": np.array([0]), # stop
    "turn_left": np.array([1]),
    "move_forward": np.array([2]),
    "turn_right": np.array([3]),
}


def check_checkpoint(args, model, optimizer, lr_scheduler) -> int:
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = model.load_state_dict(update_model_state, strict=False)

        if 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1
            print("Resume from Epoch {}".format(resume_from_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])

    return resume_from_epoch

def save_checkpoint(model, model_path, optimizer=None, epoch: int=0, save_states: bool=False):
    if hasattr(model, 'module'):
        model = model.module
    
    state_dict = {
        "model_state_dict": model.state_dict()
    }
    if save_states:
        state_dict.update({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        })

    torch.save(state_dict, model_path)
    
    
# base training agent class for Habitat
# it contains the training and validation loop
class HabitatAgent:
    def __init__(self, args, config, nav_model):
        self.args = args
        self.config = config
        self.nav_model = nav_model
        
        self.task_sim = SceneSimulator(args=args, config=config)
        
    def train(self, criterion, step_task=False):
        # init
        finished = 0
        ml_loss, cnt_loss = 0., 0.
        count = 0
        self.task_sim.gt_step = self.task_sim.count_gt_step(step_task)
        action = 'stop'
        obs, done, info = self.task_sim.actor(action)

        if step_task:
            pos = self.config["start_pos"]
            yaw = math.degrees(self.config["start_yaw"] - 180)
            rot = transback_rotation(euler_angles_to_rotation_matrix(np.array([0, 0, yaw])))
            self.task_sim.set_state(pos, rot)

        while not(self.task_sim.episode_over):
            agent_pos = transform_position(info["agent position"])
            agent_rot = transform_rotation(info["agent rotation"])

            input = {
                'obs': obs,                     # ["left", "front", "right"] observations，if depth needed, turn to habitat_base.visualization.display_env
                'agent_position': agent_pos,    # xyz
                'agent_rotation': agent_rot,    # rotation matrix
            }

            label = self.task_sim.get_next_action(info["target coord"])
            label_onehot = torch.from_numpy(label_index[label])
            
            action = self.nav_model.nav(input)
            output = torch.from_numpy(label_index[action])

            # print gt action
            print(f"The ground truth action for step {self.task_sim.step} is {label}")
                    
            cnt_loss += criterion(output, label_onehot.to(output.device)) / self.args.gradient_accumulation_step

            ml_loss += cnt_loss.detach()
            count += 1

            if count % self.args.gradient_accumulation_step == 0:
                cnt_loss.backward()
                cnt_loss = 0.    
            
            obs, done, info = self.task_sim.actor(action)
            
        self.task_sim.close()
        return ml_loss/count if count > 0 else ml_loss, self.task_sim.return_results()


    def validate(self, step_task=False):  
        action = 'stop'
        obs, done, info = self.task_sim.actor(action)
        self.task_sim.gt_step = self.task_sim.count_gt_step(step_task)
        if step_task:
            pos = self.config["start_pos"]
            yaw = math.degrees(self.config["start_yaw"] - 180)
            rot = transback_rotation(euler_angles_to_rotation_matrix(np.array([0, 0, yaw])))
            self.task_sim.set_state(pos, rot)
        
        while not(self.task_sim.episode_over):
            agent_pos = transform_position(info["agent position"])
            agent_rot = transform_rotation(info["agent rotation"])

            input = {
                'obs': obs,                     # ["left", "front", "right"] observations，if depth needed, turn to habitat_base.visualization.display_env
                'agent_position': agent_pos,    # xyz
                'agent_rotation': agent_rot,    # rotation matrix
            }

            label = self.task_sim.get_next_action(info["target coord"])           
            action = self.nav_model.nav(input)

            # print gt action
            print(f"The ground truth action for step {self.task_sim.step} is {label}")                    
            
            obs, done, info = self.task_sim.actor(action)

        self.task_sim.close()
        return self.task_sim.return_results()
