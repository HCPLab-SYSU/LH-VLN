import habitat_sim
import magnum as mn
import numpy as np
import math
import quaternion
import operator
import os
import json
from typing import List, Optional, Tuple, Union
from .visualization import display_env
from .config import make_setting, make_cfg
# @title Define Simulation Utility Functions { display-mode: "form" }
# @markdown (double click to show code)

# @markdown - remove_all_objects
# @markdown - simulate
# @markdown - sample_object_state

class SceneSimulator:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.scene = config['Scene']            # scene file
        self.robot = config['Robot']            # robot type
        self.target = config['Object']          # target object
        self.region = config['Region']          # region of target object
        self.ins = config['Task instruction']   # task instruction
        # self.task = config['Subtask list']      # subtask list

        self.save_path = None # os.path.join(args.test_task_data, 'batch_1', str(len(self.target)), self.ins)

        # init simulator
        self.sim_settings = make_setting(self.args, self.scene, self.robot)
        self.cfg = make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)

        # Pathfinder
        self.pathfinder = self.sim.pathfinder
        # init the agent with navigable point
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        # Set agent state
        agent_state = habitat_sim.AgentState()
        # sample the navigable point as agent's initial position
        sample_navigable_point = self.pathfinder.get_random_navigable_point()
        agent_state.position = sample_navigable_point - np.array([0, 0, -0.25])  # in world space
        self.agent.set_state(agent_state)

        # init path follower
        self.action_space = self.cfg.agents[self.sim_settings["default_agent"]].action_space
        self.follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder = self.pathfinder,
            agent = self.agent,
            goal_radius = self.args.success_dis,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        # init obs
        self.observations = self.sim.step("move_forward")

        # metrics
        self.step = -1
        self.stage = 0
        self.nav_steps = []
        self.successes = []  
        self.oracle_successes = []  
        for i in range(len(self.target)):
            self.successes.append(False)
            self.oracle_successes.append(False)
        self.nav_errors = []

        self.info = self.get_info()  # get the info of the current state
        self.target_num = len(self.target)  # number of target objects
        self.gt_step = None  # ground truth path steps
        self.gt_path = [self.info['geo dis']]  # ground truth path length
        
        self.done = False
        self.episode_over = False

    def actor(self, action):
        """
        Perform base action
        Args:
            action: the action to be taken
        Returns:
            obs: the visual observations [left, front, right]
            done: whether all the stage is over
            info: the info of the current state. The navigation model should only use the agent state in the info
        """
        if action == "stop":
            pass
        else:
            self.observations = self.sim.step(action)    # obtain visual observations
            
        obj_target = self.target[self.stage]   # get the target object
        obs = display_env(self.observations, action, self.save_path, self.step, obj_target)

        if self.step == -1:
            self.step += 1
            return obs, self.done, self.info
        else:
            print("action: %s, step: %d" % (action, self.step))

        self.step += 1

        self.info = self.get_info()  # get the info of the current state
        if self.info['geo dis'] < self.args.success_dis:
            self.oracle_successes[self.stage] = True

        if action == "stop":    
            if self.info['geo dis'] < self.args.success_dis:  # if the agent is close to the target object
                self.successes[self.stage] = True
                print("\n***** nav to %s success! *****\n" % obj_target)
            else:
                print("\n***** nav to %s fail! *****\n" % obj_target)
            self.nav_errors.append(self.info['geo dis'])

            if len(self.nav_steps) == 0:
                self.nav_steps.append(self.step)
            else:
                former = sum(self.nav_steps)  # sum of the previous steps
                self.nav_steps.append(self.step-former)

            self.stage += 1
            if self.stage >= self.target_num:  # if all target objects are reached
                self.done = True
                self.episode_over = True
                print("\n***** navigation over! *****\n")
                return obs, self.done, self.info

            self.info = self.get_info()  # get the info of the current state
            self.gt_path.append(self.info['geo dis'])  # ground truth path length

        if self.step >= self.args.max_step:  # if the maximum step is reached
            self.episode_over = True
            print("\n***** maximum step reached! *****\n")
            if len(self.nav_steps) == 0:
                self.nav_steps.append(self.step)
            else:
                former = sum(self.nav_steps)  # sum of the previous steps
                self.nav_steps.append(self.step-former)
            self.nav_errors.append(self.info['geo dis'])

        return obs, self.done, self.info
    
    def set_state(self, pos, rot):
        """
        Args:
            pos: the position of the agent (-y, z, -x)
            rot: the oritation of the agent (w, -y, z, -x)
        Returns:
            None
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(pos)  # in world space
        if isinstance(rot, quaternion.quaternion):
            agent_state.rotation = rot
        else:
            agent_state.rotation = quaternion.quaternion(rot[0], rot[1], rot[2], rot[3])
        self.agent.set_state(agent_state)


    def obj_count(self):
        obj_num = 0
        useless = ["wall", "frame", "floor", "sheet", "Unknown", "stairs", "unknown",
                    "ceiling", "window", "curtain", "pillow", "beam", "decoration"]
        scene = self.sim.semantic_scene
        for region in scene.regions:
            for obj in region.objects:
                obj_id = obj.id.split("_")[1]
                if any(c in obj.category.name() for c in useless):
                    continue
                else:
                    obj_num += 1
        return obj_num

    def print_scene_recur(self, file):
        """
        Args:
            file: the file name
        Returns:
            None
        """
        out_path = "nav_gen/data/gen_data/scene/" + file
        useless = ["wall", "frame", "floor", "sheet", "Unknown", "stairs", "unknown",
                    "ceiling", "window", "curtain", "pillow", "beam", "decoration"]
        # if not os.path.exists(out_path):
        #     os.makedirs(out_path)
        scene = self.sim.semantic_scene
        print(
            f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
        )
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
        for region in scene.regions:
            print(
                f"Region id:{region.id},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            with open(out_path + ".txt",'a') as f:
                f.write(
                    f"Region id:{region.id},"
                    f" position:{region.aabb.center}"
                    f"\n"
                )
            for obj in region.objects:
                obj_id = obj.id.split("_")[1]
                print(
                    f"Object id:{obj_id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                if any(c in obj.category.name() for c in useless):
                    continue
                else:
                    with open(out_path + ".txt",'a') as f:
                        f.write(
                            f"Id:{obj_id}, name:{obj.category.name()},"
                            f" position:{[obj.aabb.center[0], obj.aabb.center[1], obj.aabb.center[2]]}"
                            f"\n"
                        )

    def get_coord(self, obj_target):
        """
        Return the coord of the target object
        Args:
            obj_target: the target object
        Returns:
            coord_list: the list of the coordinates of the target object
        """
        scene = self.sim.semantic_scene
        coord_list = []
        index = self.target.index(obj_target)
        region_id = self.region[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    coord_list.append(obj.aabb.center)
        
        if coord_list == []:
            print("wrong target")
            return 0
        else:
            return coord_list

    def target_dis(self, coord_list):
        """
        Return the cloest object and the distance(Euler distance)
        Args:
            coord_list: the list of the coordinates of the target object
        Returns:
            min_dis: the distance between agent and target
            coord: the coordinate of the target object
        """
        agent_position, _ = self.return_state()
        # compute the distance between agent and target
        dis = [math.dist(np.roll(agent_position, 1)[:-1], np.roll(coord, 1)[:-1]) for coord in coord_list]
        index, min_dis = min(enumerate(dis), key=operator.itemgetter(1))
        return min_dis, coord_list[index]

    def geodesic_distance(
        self,
        position_b_list
    ) -> float:
        position_a, _ = self.return_state()
        # print(self.pathfinder.is_navigable(position_a))
        # print(self.pathfinder.is_navigable(position_b))
        geo_dis = math.inf
        coord = position_b_list[0]
        
        for position_b in position_b_list:
            path = habitat_sim.nav.ShortestPath()
            path.requested_end = np.array(
                np.array(position_b, dtype=np.float32)
            )

            path.requested_start = np.array(position_a, dtype=np.float32)

            if self.pathfinder.find_path(path):
                if path.geodesic_distance < geo_dis:
                    geo_dis = path.geodesic_distance
                    coord = position_b

        return geo_dis, coord

    # return ground path length for a task
    def count_gt_step(self, step):
        if step:
            return [self.config["end"] - self.config["start"] + 1]
        else:
            if self.args.episode_data:
                return self.config['gt_step']
            
            data_path = self.args.task_data
            action_path = os.path.join(data_path + self.config['Batch'], str(self.target_num), self.ins, 'success/trial_1')
            actions = os.listdir(action_path)
            gt = {}
            for action in actions:
                if action.split('_')[-1] not in gt:
                    gt[action.split('_')[-1]] = 1
                else:
                    gt[action.split('_')[-1]] += 1
            gt_step = []
            for key in self.target:
                gt_step.append(gt[key]-1)
            return gt_step
    
    def get_info(self):
        """
        Return the info of the current state

        returns:
            target: the target object
            target coord: the coordinate of the target object
            agent position: the position of the agent (-y,z,-x)
            agent rotation: the rotation of the agent (w, -y, z, -x)
            geo dis: the geodesic distance between agent and target
        """
        obj_target = self.target[self.stage]
        coord_list = self.get_coord(obj_target)
        if not(coord_list):
            return None, None, None, None, math.inf
        snap_coord_list = [self.pathfinder.snap_point(coord) for coord in coord_list]
        geo_dis, snap_coord = self.geodesic_distance(snap_coord_list)

        posotion, rotation = self.return_state()

        print("target coord: ", snap_coord)       
        print("agent_state: position ", posotion, ", rotation ", rotation)
        print("%f meters from the %s" % (geo_dis, obj_target))

        return {
            "target": obj_target,
            "target coord": snap_coord,
            "agent position": posotion,
            "agent rotation": rotation,
            "geo dis": geo_dis
        }
    
    def get_next_action(self, goal_pos) -> Optional[Union[int, np.ndarray]]:
        """Returns the next action along the shortest path."""
        assert self.follower is not None
        next_action = self.follower.next_action_along(goal_pos)
        return next_action
    
    def return_state(self):
        agent_state = self.agent.get_state()
        return agent_state.position, agent_state.rotation

    def return_results(self):
        """
        Return the results of the navigation task
        Returns:
            successes: whether the navigation to each target object is successful
            oracle_successes: whether the navigation to each target object is successful in oracle mode
            nav_steps: the number of steps taken to reach each target object
            nav_errors: the distance between agent and target object when the agent stops
            gt_step: ground truth path steps
            gt_path: ground truth path length
        """
        return {
            "successes": self.successes,
            "oracle_successes": self.oracle_successes,
            "navigation_steps": self.nav_steps,
            "navigation_errors": self.nav_errors,
            "gt_step": self.gt_step,
            "gt_path": self.gt_path
        }
    
    def close(self):
        """Close the simulator."""
        self.sim.close()

