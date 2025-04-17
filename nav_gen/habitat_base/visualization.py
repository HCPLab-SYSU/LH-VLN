from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import shutil
import os
import PyQt5
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir


def display_env(observations, action, save_path, step, obj_target):
    """
    Display the RGB image of the agent.
    """
    rgb_r = observations["color_sensor_r"]
    rgb_l = observations["color_sensor_l"]
    rgb_f = observations["color_sensor_f"]
    # rgb_3rd = observations["color_sensor_3rd"]

    rgb_img_r = Image.fromarray(rgb_r, mode="RGBA")
    rgb_img_l = Image.fromarray(rgb_l, mode="RGBA")
    rgb_img_f = Image.fromarray(rgb_f, mode="RGBA")
    # rgb_img_3rd = Image.fromarray(rgb_3rd, mode="RGBA")
    
    depth_r = observations["depth_sensor_r"]
    depth_l = observations["depth_sensor_l"]
    depth_f = observations["depth_sensor_f"]

    depth_img_r = Image.fromarray((depth_r / 10 * 255).astype(np.uint8), mode="L")
    depth_img_l = Image.fromarray((depth_l / 10 * 255).astype(np.uint8), mode="L")
    depth_img_f = Image.fromarray((depth_f / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img_l, rgb_img_f, rgb_img_r, depth_img_l, depth_img_f, depth_img_r]
    arr_new = []
    for img in arr:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        img = img.resize((366, 366))
        arr_new.append(img)
    titles = ["left", "front", "right", "depth_left", "depth_front", "depth_right"]
    arr = arr_new

    if not(save_path):
        return arr[:3]

    file = save_path + '/temp'
    if '/' in obj_target:
        obj_target = obj_target.replace('/', '')
    # file = save_path + "/success/trial_1"
    if not os.path.isdir(file):
        os.mkdir(file)
    elif os.path.isdir(file) and step == -1:
        shutil.rmtree(file)
        os.mkdir(file)
    if not os.path.isdir(file + "/" +str(step) + "_" + action + "_for_" + obj_target):
        os.mkdir(file + "/" +str(step) + "_" + action + "_for_" + obj_target)

    for i, data in enumerate(arr):
        plt.axis('off')
        plt.imshow(data)
        plt.savefig(file + "/" + str(step) + "_" + action + "_for_" + obj_target + "/" + titles[i] + ".png", bbox_inches='tight', pad_inches=0)
    
    return arr[:3]



# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), key_points=None
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        plt.imshow(data)

    # plt.show(block=False)
    plt.savefig("./test.png")