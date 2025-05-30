import imageio
from habitat_sim.utils import common as utils
from .visualization import display_sample
import magnum as mn
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import habitat_sim
import math


# @markdown Configure the map resolution:
meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown ---
# @markdown Customize the map slice height (global y coordinate):
custom_height = False  # @param {type:"boolean"}
height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)
display = True
output_path = ""

if display:
    module_path = os.path.abspath('')
    spec = importlib.util.spec_from_file_location("maps", module_path)
    maps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maps) 
# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)
    # plt.savefig("topdown_map.png")

def config_map(sim, save_path):
    # @markdown ###Configure Example Parameters:

    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    if not custom_height:
        # get bounding box minimum elevation for automatic height
        height = sim.pathfinder.get_bounds()[0][1]

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
        # This map is a 2D boolean array
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

        if display:
            # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
            hablab_topdown_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            hablab_topdown_map = recolor_map[hablab_topdown_map]
            # print("Displaying the raw map from get_topdown_view:")
            # display_map(sim_topdown_map)
            # print("Displaying the map from the Habitat-Lab maps module:")
            # # display_map(hablab_topdown_map)

            # easily save a map to file:
            save_file = save_path + '/'
            map_filename = os.path.join(save_file, "top_down_map.png")
            imageio.imsave(map_filename, hablab_topdown_map)

# @markdown ## Querying the NavMesh
def query_map(sim):
    # @markdown ###Query Example Parameters:
    # @markdown ---
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
        print("NavMesh area = " + str(sim.pathfinder.navigable_area))
        print("Bounds = " + str(sim.pathfinder.get_bounds()))

        # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
        pathfinder_seed = 1  # @param {type:"integer"}
        sim.pathfinder.seed(pathfinder_seed)
        nav_point = sim.pathfinder.get_random_navigable_point()
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

        # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
        # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
        print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

        # @markdown The closest boundary point can also be queried (within some radius).
        max_search_radius = 2.0  # @param {type:"number"}
        print(
            "Distance to obstacle: "
            + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
        )
        hit_record = sim.pathfinder.closest_obstacle_surface_point(
            nav_point, max_search_radius
        )
        print("Closest obstacle HitRecord:")
        print(" point: " + str(hit_record.hit_pos))
        print(" normal: " + str(hit_record.hit_normal))
        print(" distance: " + str(hit_record.hit_dist))

        vis_points = [nav_point]

        # HitRecord will have infinite distance if no valid point was found:
        if math.isinf(hit_record.hit_dist):
            print("No obstacle found within search radius.")
        else:
            # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
            perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
            print("Perturbed point : " + str(perturbed_point))
            print(
                "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
            )
            snapped_point = sim.pathfinder.snap_point(perturbed_point)
            print("Snapped point : " + str(snapped_point))
            print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
            vis_points.append(snapped_point)

        # @markdown ---
        # @markdown ### Visualization
        # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlaid.
        meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

        if display:
            xy_vis_points = convert_points_to_topdown(
                sim.pathfinder, vis_points, meters_per_pixel
            )
            # use the y coordinate of the sampled nav_point for the map height slice
            top_down_map = maps.get_topdown_map(
                sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            top_down_map = recolor_map[top_down_map]
            print("\nDisplay the map with key_point overlay:")
            display_map(top_down_map, key_points=xy_vis_points)
        
# @markdown ## Pathfinding Queries on NavMesh

# @markdown The shortest path between valid points on the NavMesh can be queried as shown in this example.

# @markdown With a valid PathFinder instance:
def pathfinding_map(sim, point):
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        seed = 4  # @param {type:"integer"}
        sim.pathfinder.seed(seed)
        # fmt off
        # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
        # fmt on
        sample1 = point[0]
        sample2 = point[1]

        # @markdown 2. Use ShortestPath module to compute path between samples.
        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = path.points
        # @markdown - Success, geodesic path length, and 3D points can be queried.
        print("found_path : " + str(found_path))
        print("geodesic_distance : " + str(geodesic_distance))
        print("path_points : " + str(path_points))

        # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min
            if display:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                top_down_map = recolor_map[top_down_map]
                grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                # convert world trajectory points to maps module grid points
                trajectory = [
                    maps.to_grid(
                        path_point[2],
                        path_point[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
                    for path_point in path_points
                ]
                grid_tangent = mn.Vector2(
                    trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                )
                path_initial_tangent = grid_tangent / grid_tangent.length()
                initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                # draw the agent and trajectory on the map
                maps.draw_path(top_down_map, trajectory)
                maps.draw_agent(
                    top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                )
                print("\nDisplay the map with agent and path overlay:")
                display_map(top_down_map)

            # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
            display_path_agent_renders = False  # @param{type:"boolean"}
            if display_path_agent_renders:
                print("Rendering observations at path points:")
                tangent = path_points[1] - path_points[0]
                agent_state = habitat_sim.AgentState()
                for ix, point in enumerate(path_points):
                    if ix < len(path_points) - 1:
                        tangent = path_points[ix + 1] - point
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            point, point + tangent, np.array([0, 1.0, 0])
                        )
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        sim.agent.set_state(agent_state)

                        observations = sim.get_sensor_observations()
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]

                        if display:
                            display_sample(rgb, semantic, depth)
