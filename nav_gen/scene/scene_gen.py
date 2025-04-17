import csv
import habitat_sim
import math
import magnum as mn
import numpy as np
import operator
import os


def read_scene(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        title = next(reader)[13:-1]

        scene = []
        for row in reader:
            count_list = row[13:-1]

            indices = [(int(item), index) for index, item in enumerate(count_list)]
            sorted_list = sorted(indices, key=lambda x: x[0], reverse=True)
            max_indices = [pair[1] for pair in sorted_list[:1]]
            
            scene.append(row[0] + '_' + '_'.join([title[i] for i in max_indices]))

    return scene

def make_setting(scene_file):
    if int(scene_file[2]) < 8:
        split = 'train/'
    else:
        split = 'val/'
    test_scene = 'data/scene_datasets/hm3d/' + split + scene_file
    scene_dataset = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    height = 1
    sim_settings = {
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "scene": test_scene,  # Scene path
        "scene_dataset": scene_dataset,
        "default_agent": 0,
        "sensor_height": height,  # Height of sensors in meters
        "color_sensor_f": rgb_sensor,  # RGB sensor
        "color_sensor_l": rgb_sensor,  # RGB sensor
        "color_sensor_r": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }
    return sim_settings

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    if "scene_dataset" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor_f": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "color_sensor_l": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "color_sensor_r": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_specs.append(sensor_spec)
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "stop": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def print_scene_recur(sim, file):
    out_path = file
    useless = ["wall", "frame", "floor", "sheet", "Unknown", "stairs", "unknown",
                "ceiling", "window", "curtain", "pillow", "beam", "decoration"]
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    scene = sim.semantic_scene
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

file = 'Per_Scene_Total_Weighted_Votes.csv'
scene = read_scene(file)
# print(scene)
for i in range(len(scene)):
    l = scene[i].split('_')
    print(l)
    file = l[0]

    sim_settings = make_setting(file)
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    print_scene_recur(sim, file)
    sim.close()