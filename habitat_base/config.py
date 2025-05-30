import habitat_sim
import math
import magnum as mn

def make_setting(args, scene_file, robot):
    # test_scene = "102343992"
    if int(scene_file[2]) < 8:
        split = 'train/'
    else:
        split = 'val/'
    test_scene = args.scene + split + scene_file #+ '/' + scene_file.split('-')[-1] + '.semantic.glb'
    scene_dataset = args.scene_dataset

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    if robot == 'spot':
        height = 0.5
    else:
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
        "color_sensor_3rd": rgb_sensor,  # RGB sensor
        "depth_sensor_f": depth_sensor,  # depth sensor
        "depth_sensor_l": depth_sensor,  # depth sensor
        "depth_sensor_r": depth_sensor,  # depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }
    return sim_settings

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 1
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    if "scene_dataset" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor_f": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "color_sensor_l": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "color_sensor_r": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "color_sensor_3rd": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(
                0.0,
                settings["sensor_height"] + 0.5,
                1.0,
            ),
            "orientation": [-math.pi / 4, 0.0, 0.0],
        },
        "depth_sensor_l": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "depth_sensor_f": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_sensor_r": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
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
            if sensor_uuid == "color_sensor_3rd":
                sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
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
        # "stop": habitat_sim.agent.ActionSpec("stop"),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])