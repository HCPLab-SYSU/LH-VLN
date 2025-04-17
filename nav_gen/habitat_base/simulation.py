import habitat_sim
import magnum as mn
import numpy as np
import math
import operator
import random
import os
from typing import List, Optional, Tuple, Union
from .visualization import display_env
from .config import make_setting, make_cfg


class SceneSimulator:
    def __init__(self, args, config):
        self.args = args
        self.scene = config['Scene']            # scene file
        self.robot = config['Robot']            # robot type
        self.target = config['Object']          # target object
        self.region = config['Region']          # region of target object
        self.ins = config['Task instruction']   # task instruction
        # self.task = config['Subtask list']      # subtask list

        self.save_path = args.task_path + str(len(self.target)) + '/' + self.ins

        # init simulator
        self.sim_settings = make_setting(self.args, self.scene, self.robot)
        self.cfg = make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)

        # Managers of various Attributes templates
        self.obj_attr_mgr = self.sim.get_object_template_manager()
        self.prim_attr_mgr = self.sim.get_asset_template_manager()
        self.stage_attr_mgr = self.sim.get_stage_template_manager()
        # Manager providing access to rigid objects
        self.rigid_obj_mgr = self.sim.get_rigid_object_manager()
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
        # init yaw
        self.yaw = 180
        # init path follower
        self.action_space = self.cfg.agents[self.sim_settings["default_agent"]].action_space
        self.follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder = self.pathfinder,
            agent = self.agent,
            goal_radius = 1.0,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        # init obs
        self.observations = self.sim.step("move_forward")

    def actor(self, action, step, success):
        """
        Perform base action
        Args:
            action: the action to be taken
            step: the step number
            success: the number of finished subtargets
        Returns:
            images: the visual observations
        """
        if action == "stop":
            pass
        else:
            self.observations = self.sim.step(action)    # obtain visual observations
        # adjust the angle based on the action
        if action == "move_forward" or action == "stop":
            pass
        elif action == "turn_left":
            self.yaw += 30
        elif action == "turn_right":
            self.yaw -= 30
        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw <= -180:
            self.yaw += 360
        if step is not None or step != -1:
            print("action: %s, step: %d" % (action, step))
        obj_target = self.target[success]   # get the target object
        images = display_env(self.observations, action, self.save_path, step, obj_target)
        return images
    
    def set_state(self, pos, yaw):
        """
        Args:
            pos: the position of the agent
            yaw: the yaw of the agent
        Returns:
            None
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(pos)  # in world space
        self.agent.set_state(agent_state)

        d_yaw = (yaw - self.yaw)/30
        if d_yaw > 0:
            for i in range(int(d_yaw)):
                self.observations = self.sim.step("turn_left")
        else:
            for i in range(int(-d_yaw)):
                self.observations = self.sim.step("turn_right")

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
        agent_state = self.agent.get_state()
        agent_position = agent_state.position
        # compute the distance between agent and target
        dis = [math.dist(np.roll(agent_position, 1)[:-1], np.roll(coord, 1)[:-1]) for coord in coord_list]
        index, min_dis = min(enumerate(dis), key=operator.itemgetter(1))
        return min_dis, coord_list[index]

    def geodesic_distance(
        self,
        position_b_list
    ) -> float:
        agent_state = self.agent.get_state()
        position_a = agent_state.position
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
        # return path
    
    def get_info(self, success):
        """
        Return the info of the current state
        Args:
            success: the number of finished subtargets
        Returns:
            obj_target: the target object
            coord: the coordinate of the target object
            agent_state: the position of the agent
            yaw: the yaw of the agent
            dis: the euler distance between agent and target
            geo_dis: the geodesic distance between agent and target
        """
        obj_target = self.target[success]
        coord_list = self.get_coord(obj_target)
        if not(coord_list):
            return None, None, None, None, math.inf
        snap_coord_list = [self.pathfinder.snap_point(coord) for coord in coord_list]
        geo_dis, snap_coord = self.geodesic_distance(snap_coord_list)

        agent_state = self.agent.get_state()

        print("target coord: ", snap_coord)       
        print("agent_state: position ", agent_state.position, ", yaw ", self.yaw)
        print("%f meters from the %s" % (geo_dis, obj_target))

        return obj_target, snap_coord, agent_state.position, self.yaw, geo_dis
    
    def get_next_action(self, goal_pos) -> Optional[Union[int, np.ndarray]]:
        """Returns the next action along the shortest path."""
        assert self.follower is not None
        next_action = self.follower.next_action_along(goal_pos)
        return next_action
    
    def return_state(self):
        agent_state = self.agent.get_state()
        return agent_state.position, self.yaw


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation


# sample a random valid state for the object from the scene bounding box or navmesh
def sample_object_state(
    sim, obj, from_navmesh=True, maintain_object_up=True, max_tries=100, bb=None
):
    # check that the object is not STATIC
    if obj.motion_type is habitat_sim.physics.MotionType.STATIC:
        print("sample_object_state : Object is STATIC, aborting.")
    if from_navmesh:
        if not sim.pathfinder.is_loaded:
            print("sample_object_state : No pathfinder, aborting.")
            return False
    elif not bb:
        print(
            "sample_object_state : from_navmesh not specified and no bounding box provided, aborting."
        )
        return False
    tries = 0
    valid_placement = False
    # Note: following assumes sim was not reconfigured without close
    scene_collision_margin = stage_attr_mgr.get_template_by_id(0).margin
    while not valid_placement and tries < max_tries:
        tries += 1
        # initialize sample location to random point in scene bounding box
        sample_location = np.array([0, 0, 0])
        if from_navmesh:
            # query random navigable point
            sample_location = sim.pathfinder.get_random_navigable_point()
        else:
            sample_location = np.random.uniform(bb.min, bb.max)
        # set the test state
        obj.translation = sample_location
        if maintain_object_up:
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            obj.rotation = y_rotation * obj.rotation
        else:
            # unconstrained random rotation
            obj.rotation = ut.random_quaternion()

        # raise object such that lowest bounding box corner is above the navmesh sample point.
        if from_navmesh:
            obj_node = obj.root_scene_node
            xform_bb = habitat_sim.geo.get_transformed_bb(
                obj_node.cumulative_bb, obj_node.transformation
            )
            # also account for collision margin of the scene
            obj.translation += mn.Vector3(
                0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
            )

        # test for penetration with the environment
        if not sim.contact_test(obj.object_id):
            valid_placement = True

    if not valid_placement:
        return False
    return True