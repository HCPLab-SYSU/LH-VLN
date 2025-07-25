import torch
import math
import numpy as np
from collections import defaultdict

MAX_DIST = 30
MAX_STEP = 10


def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx ** 2 + dy ** 2), 1e-8)
    xyz_dist = max(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx / xy_dist)  # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


def compute_coord(pos, rot, base_i):
    # w = rot.real
    # x, y, z = rot.imag
    # base_yaw = math.atan2(2*(w*z + x*y), 1- 2*(y**2 + z**2))
    base_yaw = rot

    realated_yaw =  -math.radians(60) + base_i * math.radians(60)
    real_yaw = base_yaw + realated_yaw

    dx = 0.25 * math.cos(real_yaw)
    dy = 0.25 * math.sin(real_yaw)

    return [pos[0] + dx, pos[1] + dy, pos[2]]

class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda: defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda: defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":  # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, start_vp):
        self.start_vp = start_vp  # start viewpoint

        self.node_positions = {}  # viewpoint to position (x, y, z)
        self.graph = FloydGraph()  # shortest path graph
        self.node_embeds = {}  # {viewpoint: feature (sum feature, count)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}  # {viewpoint: {t: prob}}
        self.node_step_ids = {}
        self.pooling_mode = 'mean'

    def update_graph(self, nav_msg):
        self.node_positions[0] = nav_msg["pos"]
        for i in range(3):
            self.node_positions[i+1] = compute_coord(nav_msg["pos"], nav_msg["rot"], i)
            dist = 0.25
            self.graph.add_edge(0, i+1, dist)
        self.graph.update(0)

    def update_node_embed(self, vp, embed, rewrite=False):
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
        else:
            if vp in self.node_embeds:
                if self.pooling_mode == "max":
                    pooling_features, _ = torch.max(torch.stack([self.node_embeds[vp][0], embed.clone()]), dim=0)
                    self.node_embeds[vp][0] = pooling_features
                elif self.pooling_mode == "mean":
                    self.node_embeds[vp][0] += embed.clone()
                else:
                    raise NotImplementedError('`pooling_mode` Only support ["mean", "max"]')
                self.node_embeds[vp][1] += 1
            else:
                self.node_embeds[vp] = [embed, 1]


    def get_node_embed(self, vp):
        if self.pooling_mode == "max":
            return self.node_embeds[vp][0]
        elif self.pooling_mode == "mean":
            return self.node_embeds[vp][0] / self.node_embeds[vp][1]
        else:
            raise NotImplementedError('`pooling_mode` Only support ["mean", "max"]')

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.graph.distance(cur_vp, vp) / MAX_DIST, \
                     len(self.graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,  # (x, y, z)
                'visited': self.graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))

        return {'nodes': nodes, 'edges': edges}

class ContinuousGraphMap(object):
    """
    Graph map for continuous navigation environments.
    Modified from ETPNav's GraphMap to work with continuous coordinates.
    """
    
    def __init__(self, loc_noise=0.5, ghost_aug=0.0, merge_ghost=True):
        self.node_pos = {}          # node_id -> (x, y, z)
        self.node_embeds = {}       # node_id -> features
        self.node_step_ids = {}     # node_id -> step_id
        
        self.ghost_cnt = 0
        self.ghost_pos = {}         # ghost_id -> list of positions
        self.ghost_mean_pos = {}    # ghost_id -> mean position
        self.ghost_embeds = {}      # ghost_id -> [accumulated_features, count]
        self.ghost_fronts = {}      # ghost_id -> list of front nodes
        
        self.loc_noise = loc_noise
        self.ghost_aug = ghost_aug
        self.merge_ghost = merge_ghost
        
        # Graph connectivity (simple adjacency)
        self.adjacency = defaultdict(list)
        self.distances = defaultdict(dict)
        
        # Current node counter
        self.node_counter = 0

        self.start_pos = None
        
    def _localize(self, pos, pos_dict, ignore_height=False):
        """Find the closest node within loc_noise distance"""
        min_dis = float('inf')
        min_node = None
        
        for node_id, node_pos in pos_dict.items():
            if ignore_height:
                dis = np.sqrt((pos[0] - node_pos[0])**2 + (pos[2] - node_pos[2])**2)
            else:
                dis = np.sqrt(np.sum((np.array(pos) - np.array(node_pos))**2))
            
            if dis < min_dis:
                min_dis = dis
                min_node = node_id
                
        return min_node if min_dis < self.loc_noise else None
    
    def add_node(self, pos, features, step_id):
        """Add a new node to the graph"""
        # Check if position is close to existing nodes
        existing_node = self._localize(pos, self.node_pos)

        # Record the start position on first call
        if self.start_pos is None:
            self.start_pos = pos
        
        if existing_node is not None:
            # Update existing node
            self.node_embeds[existing_node] = features
            self.node_step_ids[existing_node] = step_id
            return existing_node
        else:
            # Create new node
            node_id = f"n{self.node_counter}"
            self.node_counter += 1
            
            self.node_pos[node_id] = pos
            self.node_embeds[node_id] = features
            self.node_step_ids[node_id] = step_id
            
            return node_id
    
    def add_edge(self, node1, node2, distance):
        """Add an edge between two nodes"""
        if node2 not in self.adjacency[node1]:
            self.adjacency[node1].append(node2)
            self.adjacency[node2].append(node1)
            self.distances[node1][node2] = distance
            self.distances[node2][node1] = distance
    
    def get_neighbors(self, node_id):
        """Get all neighbors of a node"""
        return self.adjacency.get(node_id, [])
    
    def get_all_nodes(self):
        """Get all node IDs"""
        return list(self.node_pos.keys())
    
    def get_shortest_path_distance(self, start, end):
        """Simple shortest path distance (can be improved with Dijkstra)"""
        if start == end:
            return 0.0
        
        if end in self.distances[start]:
            return self.distances[start][end]
        
        # For simplicity, return Euclidean distance if no path found
        if start in self.node_pos and end in self.node_pos:
            pos1, pos2 = self.node_pos[start], self.node_pos[end]
            return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))
        
        return float('inf')