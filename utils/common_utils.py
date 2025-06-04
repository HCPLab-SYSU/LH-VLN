import logging
import os
import pickle
import math
import torch
import torch.distributed as dist
import pickle
import random
import shutil
from easydict import EasyDict
import numpy as np
import quaternion


def transform_position(position_yzx):
    """
    Convert position from (-y, z, -x) format to standard (x, y, z) format.
    
    This function handles coordinate system transformation where the input
    uses a different axis ordering than the standard Cartesian coordinates.
    
    Args:
        position_yzx (array-like): Position array in [-y, z, -x] format
    
    Returns:
        numpy.ndarray: Position array in standard [x, y, z] format
    """
    # Input format: [y, z, -x]
    # Output format: [x, y, z]
    neg_y, z, neg_x = position_yzx
    x = -neg_x  # Recover the original x coordinate
    y = -neg_y  # Recover the original y coordinate

    position_xyz = np.array([x, y, z])
    return position_xyz


def transback_position(position_xyz):
    """
    Convert position from standard (x, y, z) format to (-y, z, -x) format.
    
    This function handles coordinate system transformation where the output
    uses a different axis ordering than the standard Cartesian coordinates.
    
    Args:
        position_xyz (array-like): Position array in [x, y, z] format
    
    Returns:
        numpy.ndarray: Position array in [-y, z, -x] format
    """
    # Input format: [x, y, z]
    # Output format: [-y, z, -x]
    neg_z, neg_x, y = position_xyz
    x = -neg_x  # Recover the original x coordinate
    z = -neg_z  # Recover the original y coordinate
    
    position_yzx = np.array([x, y, z])
    return position_yzx


def transform_rotation(quaternion_wyzx):
    """
    Convert quaternion from (w, y, z, -x) format to 3x3 rotation matrix.
    
    This function transforms a quaternion with non-standard component ordering
    into a standard rotation matrix representation.
    
    Args:
        quaternion_wyzx (array-like or quaternion.quaternion): 
            Quaternion in [w, y, z, -x] format, can be either a numpy array
            or a quaternion.quaternion object
    
    Returns:
        numpy.ndarray: 3x3 rotation matrix representing the same rotation
    """
    # Input format: [w, y, z, -x]
    # Standard quaternion format: [w, x, y, z]
    
    # if hasattr(quaternion_wyzx, 'w'):  # Check if it's a quaternion.quaternion object
    #     w = quaternion_wyzx.w
    #     y = quaternion_wyzx.y
    #     z = quaternion_wyzx.z
    #     neg_x = quaternion_wyzx.x  # Assuming the object's x attribute is actually -x
    #     x = -neg_x
    # else:  # If it's an array
    w = quaternion_wyzx.w
    y = -quaternion_wyzx.x
    z = quaternion_wyzx.y
    x = -quaternion_wyzx.z  # Assuming the object's x attribute is actually -x
    
    # Create quaternion in standard format (w, x, y, z)
    q = quaternion.quaternion(w, x, y, z)
    
    # Convert to rotation matrix
    rotation_matrix = quaternion.as_rotation_matrix(q)
    
    return rotation_matrix


def transback_rotation(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to quaternion in standard (w, x, y, z) format.
    
    This function uses the Shepperd's method for robust quaternion extraction
    from rotation matrices, which avoids numerical instabilities.
    
    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        numpy.ndarray: Quaternion in [w, x, y, z] format
    """
    # Get matrix elements for clarity
    R = rotation_matrix
    
    # Calculate the trace of the matrix
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        # Standard case
        s = math.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # R[0,0] is the largest diagonal element
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        # R[1,1] is the largest diagonal element
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        # R[2,2] is the largest diagonal element
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return quaternion.quaternion(w, -y, z, -x)


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).
    
    This function extracts Euler angles using the ZYX convention (Tait-Bryan angles).
    The rotation order is: R = R_z(yaw) * R_y(pitch) * R_x(roll)
    
    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        tuple: (roll, pitch, yaw) angles in radians
    """
    # Get matrix elements for clarity
    R = rotation_matrix
    
    # Calculate pitch (rotation around Y-axis)
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Clamp to avoid numerical errors
    pitch = math.asin(sin_pitch)
    
    # Check for gimbal lock
    if abs(sin_pitch) >= 0.99999:  # cos(pitch) ≈ 0, gimbal lock case
        # In gimbal lock, we can set roll to 0 and calculate yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        # Normal case: calculate roll and yaw
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    
    return [roll, pitch, yaw]


def euler_angles_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    
    This function creates a rotation matrix using the ZYX convention (Tait-Bryan angles).
    The rotation order is: R = R_z(yaw) * R_y(pitch) * R_x(roll)
    
    Args:
        euler_angles (array-like): [roll, pitch, yaw] angles in radians
    
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    roll, pitch, yaw = euler_angles
    
    # Calculate trigonometric values
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    
    # Construct rotation matrix using ZYX convention
    # R = R_z(yaw) * R_y(pitch) * R_x(roll)
    rotation_matrix = np.array([
        [cos_yaw * cos_pitch, 
         cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
         cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
        
        [sin_yaw * cos_pitch,
         sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
         sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
        
        [-sin_pitch,
         cos_pitch * sin_roll,
         cos_pitch * cos_roll]
    ])
    
    return rotation_matrix


def rotation_matrix_to_direction(rotation_matrix):
    """
    Extract the forward direction vector from a rotation matrix.
    
    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
        numpy.ndarray: 3D unit vector [x, y, z] representing the forward direction
    """
    # In most 3D systems, the forward direction is along the negative Z-axis
    # When no rotation is applied, forward = [0, 0, -1]
    # The rotation matrix transforms this to the actual forward direction
    forward_direction = rotation_matrix @ np.array([0, 0, -1])
    
    # Normalize to ensure it's a unit vector (though it should already be)
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    
    return forward_direction


def is_object_in_fov(agent_pos, agent_rot, object_pos, fov_angle=60):
    """
    Check if an object is within the agent's field of view (FOV).

    Args:
        agent_pos (numpy.ndarray): The position of the agent (-y, z, -x).
        agent_rot (quaternion.quaternion): The rotation of the agent (w, -y, z, -x).
        object_pos (numpy.ndarray): The position of the object (-y, z, -x).
        fov_angle (float): The field of view angle in degrees.

    Returns:
        bool: True if the object is within the FOV, False otherwise.
    """
    agent_pos = transform_position(agent_pos)
    object_pos = transform_position(object_pos)
    agent_rot = transform_rotation(agent_rot)

    # Calculate the direction the agent is facing
    agent_forward = rotation_matrix_to_direction(agent_rot)

    # Calculate the vector from the agent to the object
    to_object = object_pos - agent_pos
    to_object /= np.linalg.norm(to_object)  # Normalize the vector

    # Calculate the angle between the agent's forward direction and the vector to the object
    angle = np.arccos(np.clip(np.dot(agent_forward, to_object), -1.0, 1.0))
    angle = np.degrees(angle)

    return angle < fov_angle / 2


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def summary_model(model,level=2):
    message = ""
    if level < 1:
        return message
    for name1, module1 in model.named_children():
        message += "[1] {}\n".format(name1)
        if level > 1:
            for name2, module2 in module1.named_children():
                message += "- [2] {}\n".format(name2)
                if level > 2:
                    for name3, module3 in module2.named_children():
                        message += " +++ [3] {}\n".format(name3)
                        if level > 3:
                            for name4, module4 in module3.named_children():
                                message += " +++++ [4] {}\n".format(name4)
    return message


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list
