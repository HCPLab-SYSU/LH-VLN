import torch
import numpy as np
import math
from .models.nav_model import NavModel
from .tools.optims import dist_models
from .tools.gmap import ContinuousGraphMap
from .tools.memory import forget_with_entropy


class ContinuousNav:
    def __init__(self, args, global_cfg, logger, device_id):
        # self.args, self.global_cfg, self.logger, device_id = read_args()
        self.args = args
        self.global_cfg = global_cfg
        self.logger = logger
        # Ensure device is a torch.device object
        if isinstance(device_id, (int, str)):
            self.device = torch.device(f'cuda:{device_id}' if str(device_id).isdigit() else device_id)
        else:
            self.device = device_id

        model = NavModel(self.args, self.logger, self.global_cfg.Model)
        self.model, self.optimizer, self.resume_from_epoch, self.lr_scheduler = dist_models(self.args, model, self.logger)

        # Initialize episode state
        self.history = []
        self.hist_vis = []
        self.step_count = 0

        # Graph maps for each environment
        self.gmaps = []
        self.current_nodes = []  # Current node for each environment
        self.prev_nodes = []     # Previous node for each environment

        self.max_dist = 30.0                        # Maximum distance for normalization
        
    def reset_episode(self, batch_size):
        """Reset for new episodes"""
        self.gmaps = [ContinuousGraphMap(
            loc_noise=self.args.loc_noise if hasattr(self.args, 'loc_noise') else 0.5,
            ghost_aug=self.args.ghost_aug if hasattr(self.args, 'ghost_aug') else 0.0,
            merge_ghost=self.args.merge_ghost if hasattr(self.args, 'merge_ghost') else True
        ) for _ in range(batch_size)]
        self.current_nodes = [None] * batch_size
        self.prev_nodes = [None] * batch_size

        self.history = [[] for _ in range(batch_size)]
        self.hist_vis = [[] for _ in range(batch_size)]
        self.step_count = 0
        
    def calculate_rel_pos_features(self, pos1, pos2, heading1=0.0):
        """Calculate relative position features between two positions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        
        # Calculate distances
        xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
        xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)
        
        # Calculate relative heading
        heading = np.arcsin(-dx / xy_dist)
        if dy > 0:
            heading = np.pi - heading
        heading = (heading - heading1) % (2 * np.pi)
        
        # Calculate elevation
        elevation = np.arcsin(dz / xyz_dist)
        
        # Create angle features
        angle_fts = np.array([
            np.sin(heading), np.cos(heading),
            np.sin(elevation), np.cos(elevation)
        ] * (self.args.angle_feat_size // 4), dtype=np.float32)
        
        # Distance features (normalized)
        max_dist = self.args.max_dist if hasattr(self.args, 'max_dist') else 30.0
        dist_fts = np.array([
            xyz_dist / max_dist,
            min(xyz_dist / max_dist, 1.0),
            0.0  # shortest_step placeholder
        ], dtype=np.float32)
        
        return np.concatenate([angle_fts, dist_fts]).astype(np.float32)
    
    def create_panorama_features(self, batch_obs):
        """
        Create panorama features
        """
        batch_view_img_fts = []
        batch_loc_fts = []
        batch_nav_types = []
        batch_view_lens = []
        batch_cand_vpids = []
        
        for i, obs in enumerate(batch_obs):
            # Extract view features from observation
            view_feats = obs['view_feats']  # (3, 1024) = [left, front, right] each 1024
            
            # Extract 3 views: left, front, right
            view_img_fts = []
            view_img_fts.append(view_feats[0])  # left view
            view_img_fts.append(view_feats[1])  # front view  
            view_img_fts.append(view_feats[2])  # right view
            
            # Create angle features for 3 views (left=-60°, front=0°, right=60°)
            pose = obs['pose']  # [x, y, z, heading]
            base_heading = pose[3]
            
            view_ang_fts = []
            # Left view: -60 degrees
            left_heading = base_heading - math.pi/3
            view_ang_fts.append([
                math.sin(left_heading), math.cos(left_heading),
                0.0, 1.0  # elevation sin=0, cos=1
            ])
            
            # Front view: 0 degrees  
            view_ang_fts.append([
                math.sin(base_heading), math.cos(base_heading),
                0.0, 1.0  # elevation sin=0, cos=1
            ])
            
            # Right view: +60 degrees
            right_heading = base_heading + math.pi/3
            view_ang_fts.append([
                math.sin(right_heading), math.cos(right_heading),
                0.0, 1.0  # elevation sin=0, cos=1
            ])
            
            # Create navigation types (1 for navigable candidates, 0 for non-navigable)
            nav_types = [1, 1, 1]  # All 3 views are navigable candidates
            
            # Create candidate viewpoint IDs for the 3 actions
            cand_vpids = ['turn_left', 'move_forward', 'turn_right']
            
            # Pad to match mp3d format (36 views total)
            while len(view_img_fts) < 36:
                view_img_fts.append(np.zeros(1024, dtype=np.float32))
                view_ang_fts.append([0.0, 0.0, 0.0, 0.0])
                nav_types.append(0)
            
            # Create location features (angle + box features)
            view_img_fts = np.stack(view_img_fts, 0).astype(np.float32)
            view_ang_fts = np.stack(view_ang_fts, 0).astype(np.float32)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1).astype(np.float32)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts).to(self.device))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts).to(self.device))
            batch_nav_types.append(torch.LongTensor(nav_types).to(self.device))
            batch_view_lens.append(len(view_img_fts))
            batch_cand_vpids.append(cand_vpids)
        
        # Pad sequences
        max_len = max(len(fts) for fts in batch_view_img_fts)
        
        # Pad view features
        padded_view_img_fts = []
        padded_loc_fts = []
        for i in range(len(batch_obs)):
            view_fts = batch_view_img_fts[i]
            loc_fts = batch_loc_fts[i]
            
            pad_size = max_len - view_fts.shape[0]
            if pad_size > 0:
                view_pad = torch.zeros(pad_size, view_fts.shape[1], dtype=torch.float32, device=self.device)
                loc_pad = torch.zeros(pad_size, loc_fts.shape[1], dtype=torch.float32, device=self.device)
                
                padded_view_img_fts.append(torch.cat([view_fts, view_pad], 0))
                padded_loc_fts.append(torch.cat([loc_fts, loc_pad], 0))
            else:
                padded_view_img_fts.append(view_fts)
                padded_loc_fts.append(loc_fts)
        
        # Pad navigation types
        from torch.nn.utils.rnn import pad_sequence
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0)
        
        return {
            'view_img_fts': torch.stack(padded_view_img_fts),
            'loc_fts': torch.stack(padded_loc_fts),
            'nav_types': batch_nav_types,
            'view_lens': torch.LongTensor(batch_view_lens).to(self.device),
            'cand_vpids': batch_cand_vpids,
        }
    
    def update_graph(self, batch_obs, step_id, pano_embeds):
        """Update graph with new observations using panorama embeddings"""
        batch_size = len(batch_obs)
        
        for i, obs in enumerate(batch_obs):
            pose = obs['pose']  # [x, y, z, heading]
            current_pos = tuple(pose[:3])
            
            # Average the panorama embeddings to get node embedding
            obs_features = torch.mean(pano_embeds[i], dim=0)  # Average across views
            
            # Add current position as a node
            current_node = self.gmaps[i].add_node(current_pos, obs_features.detach(), step_id)
            
            # Connect to previous node if exists
            if self.prev_nodes[i] is not None:
                prev_pos = self.gmaps[i].node_pos[self.prev_nodes[i]]
                distance = np.sqrt(np.sum((np.array(current_pos) - np.array(prev_pos))**2))
                self.gmaps[i].add_edge(self.prev_nodes[i], current_node, distance)
            
            # Update current and previous nodes
            self.prev_nodes[i] = self.current_nodes[i]
            self.current_nodes[i] = current_node
    
    def prepare_navigation_inputs(self, batch_obs, history, hist_vis, pano_embeds):
        """Prepare inputs for navigation model"""
        batch_size = len(batch_obs)
        
        # Prepare global map inputs (gmap)
        gmap_vpids = []
        gmap_step_ids = []
        gmap_img_embeds = []
        gmap_pos_fts = []
        gmap_visited_masks = []
        gmap_masks = []
        
        # Prepare viewpoint inputs (vp) - actions
        vp_img_embeds = []
        vp_pos_fts = []
        vp_nav_masks = []
        vp_cand_vpids = []
        pano_masks = []
        
        for i, obs in enumerate(batch_obs):
            pose = obs['pose']  # [x, y, z, heading]
            current_pos = tuple(pose[:3])
            current_heading = pose[3]
            
            gmap = self.gmaps[i]
            
            # === Global Map (gmap) preparation ===
            # Get all graph nodes
            all_nodes = gmap.get_all_nodes()
            
            # Add stop action + all graph nodes
            if self.args.enc_full_graph:
                node_vpids = ['stop', 'turn_left', 'move_forward', 'turn_right'] + all_nodes
                node_step_ids = [self.step_count, self.step_count, self.step_count, self.step_count] + [gmap.node_step_ids.get(node, 0) for node in all_nodes]
            else:
                node_vpids = ['stop', 'turn_left', 'move_forward', 'turn_right']
                node_step_ids = [self.step_count, self.step_count, self.step_count, self.step_count]
            
            # Node embeddings
            node_embeds = []
            hidden_size = pano_embeds.shape[-1]  # Get hidden size
            for j in range(4):
                if j == 0:  # stop
                    node_embeds.append(torch.zeros(hidden_size, dtype=torch.float32, device=self.device))
                else:  # turn_left, move_forward, turn_right
                    view_idx = j - 1  # 0:left, 1:front, 2:right
                    if view_idx < pano_embeds[i].shape[0]:
                        node_embeds.append(pano_embeds[i][view_idx])
                    else:
                        node_embeds.append(torch.zeros(hidden_size, dtype=torch.float32, device=self.device))
            
            if self.args.enc_full_graph:
                # Graph node embeddings
                for node in all_nodes:
                    node_embeds.append(gmap.node_embeds[node])
            
            # Position features for graph nodes
            node_pos_fts = []
            for vpid in node_vpids:
                if vpid == 'stop':
                    # Stop position (same as current)
                    pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                elif vpid == 'turn_left':
                    pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                elif vpid == 'move_forward':
                    # Calculate forward position
                    move_dist = 0.25
                    new_x = current_pos[0] + move_dist * np.sin(current_heading)
                    new_y = current_pos[1] + move_dist * np.cos(current_heading)
                    forward_pos = (new_x, new_y, current_pos[2])
                    pos_ft = self.calculate_rel_pos_features(current_pos, forward_pos, current_heading)
                elif vpid == 'turn_right':
                    pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                else:
                    # Graph node position
                    node_pos = gmap.node_pos[vpid]
                    pos_ft = self.calculate_rel_pos_features(current_pos, node_pos, current_heading)
                node_pos_fts.append(pos_ft)
            
            # Visited masks for graph nodes
            node_visited_masks = []
            for vpid in node_vpids:
                if vpid in ['stop', 'turn_left', 'move_forward', 'turn_right']:
                    node_visited_masks.append(False)  # Stop is never visited
                else:
                    # Mark as visited for history node
                    is_visited = True
                    node_visited_masks.append(is_visited)
            
            # === Viewpoint (vp) preparation - Actions ===
            # Use panorama embeddings for action candidates
            # pano_embeds[i] should be [num_views, hidden_size]
            
            # Add stop action at the beginning
            action_embeds = [torch.zeros(hidden_size, dtype=torch.float32, device=self.device)]  # Stop action
            action_embeds.extend([pano_embeds[i][j] for j in range(min(3, pano_embeds[i].shape[0]))])  # 3 action views
            
            # Action position features
            action_vpids = ['stop', 'turn_left', 'move_forward', 'turn_right']

            start_pos = gmap.start_pos if gmap.start_pos is not None else current_pos
            cur_start_pos_fts = self.calculate_rel_pos_features(current_pos, start_pos, current_heading)
            action_cand_pos_fts = []
            
            for j, action_name in enumerate(action_vpids):
                if action_name == 'stop':
                    cand_pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                elif action_name == 'turn_left':
                    cand_pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                elif action_name == 'move_forward':
                    # Calculate forward position
                    move_dist = 0.25
                    new_x = current_pos[0] + move_dist * np.sin(current_heading)
                    new_y = current_pos[1] + move_dist * np.cos(current_heading)
                    forward_pos = (new_x, new_y, current_pos[2])
                    cand_pos_ft = self.calculate_rel_pos_features(current_pos, forward_pos, current_heading)
                elif action_name == 'turn_right':
                    cand_pos_ft = self.calculate_rel_pos_features(current_pos, current_pos, current_heading)
                
                action_cand_pos_fts.append(cand_pos_ft)
            
            num_actions = len(action_vpids)
            vp_pos_fts_14d = np.zeros((num_actions, 14), dtype=np.float32)
            vp_pos_fts_14d[:, :7] = cur_start_pos_fts
            for j in range(num_actions):
                vp_pos_fts_14d[j, 7:] = action_cand_pos_fts[j]
            
            # Store results
            gmap_vpids.append(node_vpids)
            gmap_step_ids.append(torch.tensor(node_step_ids, dtype=torch.long, device=self.device))
            gmap_img_embeds.append(torch.stack(node_embeds))
            gmap_pos_fts.append(torch.tensor(np.array(node_pos_fts), dtype=torch.float32, device=self.device))
            gmap_visited_masks.append(torch.tensor(node_visited_masks, dtype=torch.bool, device=self.device))
            gmap_masks.append(torch.ones(len(node_vpids), dtype=torch.bool, device=self.device))
            
            vp_img_embeds.append(torch.stack(action_embeds))
            vp_pos_fts.append(torch.tensor(vp_pos_fts_14d, dtype=torch.float32, device=self.device))
            vp_nav_masks.append(torch.ones(len(action_vpids), dtype=torch.bool, device=self.device))
            vp_cand_vpids.append(action_vpids)
            pano_masks.append(torch.ones(len(action_vpids), dtype=torch.bool, device=self.device))
        
        # Pad sequences
        max_gmap_len = max(len(vpids) for vpids in gmap_vpids)
        max_vp_len = max(len(vpids) for vpids in vp_cand_vpids)
        
        # Pad gmap features
        padded_gmap_step_ids = []
        padded_gmap_img_embeds = []
        padded_gmap_pos_fts = []
        padded_gmap_visited_masks = []
        padded_gmap_masks = []
        
        for i in range(batch_size):
            step_ids = gmap_step_ids[i]
            padded_step_ids = torch.cat([step_ids, torch.zeros(max_gmap_len - len(step_ids), dtype=torch.long, device=self.device)])
            padded_gmap_step_ids.append(padded_step_ids)
            
            img_embeds = gmap_img_embeds[i]
            pad_size = max_gmap_len - img_embeds.shape[0]
            padded_img_embeds = torch.cat([img_embeds, torch.zeros(pad_size, hidden_size, dtype=torch.float32, device=self.device)])
            padded_gmap_img_embeds.append(padded_img_embeds)
            
            pos_fts = gmap_pos_fts[i]
            pad_size = max_gmap_len - pos_fts.shape[0]
            padded_pos_fts = torch.cat([pos_fts, torch.zeros(pad_size, pos_fts.shape[1], dtype=torch.float32, device=self.device)])
            padded_gmap_pos_fts.append(padded_pos_fts)
            
            visited_masks = gmap_visited_masks[i]
            padded_visited_masks = torch.cat([visited_masks, torch.zeros(max_gmap_len - len(visited_masks), dtype=torch.bool, device=self.device)])
            padded_gmap_visited_masks.append(padded_visited_masks)
            
            masks = gmap_masks[i]
            padded_masks = torch.cat([masks, torch.zeros(max_gmap_len - len(masks), dtype=torch.bool, device=self.device)])
            padded_gmap_masks.append(padded_masks)
        
        # Pad vp features
        padded_vp_img_embeds = []
        padded_vp_pos_fts = []
        padded_vp_nav_masks = []
        padded_pano_masks = []
        
        for i in range(batch_size):
            img_embeds = vp_img_embeds[i]
            pad_size = max_vp_len - img_embeds.shape[0]
            padded_img_embeds = torch.cat([img_embeds, torch.zeros(pad_size, hidden_size, dtype=torch.float32, device=self.device)])
            padded_vp_img_embeds.append(padded_img_embeds)
            
            pos_fts = vp_pos_fts[i]
            pad_size = max_vp_len - pos_fts.shape[0]
            padded_pos_fts = torch.cat([pos_fts, torch.zeros(pad_size, pos_fts.shape[1], dtype=torch.float32, device=self.device)])
            padded_vp_pos_fts.append(padded_pos_fts)
            
            nav_masks = vp_nav_masks[i]
            padded_nav_masks = torch.cat([nav_masks, torch.zeros(max_vp_len - len(nav_masks), dtype=torch.bool, device=self.device)])
            padded_vp_nav_masks.append(padded_nav_masks)
            
            pano_mask = pano_masks[i]
            padded_pano_mask = torch.cat([pano_mask, torch.zeros(max_vp_len - len(pano_mask), dtype=torch.bool, device=self.device)])
            padded_pano_masks.append(padded_pano_mask)
        
        # Create final batch dictionary
        batch_dict = {
            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': torch.stack(padded_gmap_step_ids),
            'gmap_img_embeds': torch.stack(padded_gmap_img_embeds),
            'gmap_pos_fts': torch.stack(padded_gmap_pos_fts),
            'gmap_visited_masks': torch.stack(padded_gmap_visited_masks),
            'gmap_masks': torch.stack(padded_gmap_masks),
            'gmap_pair_dists': torch.zeros(batch_size, max_gmap_len, max_gmap_len, dtype=torch.float32, device=self.device),  # Placeholder
            
            'vp_img_embeds': torch.stack(padded_vp_img_embeds),
            'vp_pos_fts': torch.stack(padded_vp_pos_fts),
            'vp_nav_masks': torch.stack(padded_vp_nav_masks),
            'vp_cand_vpids': vp_cand_vpids,
            'pano_masks': torch.stack(padded_pano_masks),
            
            'instruction': [batch_obs[i]['instruction'] for i in range(batch_size)],
            'history': history,
            'hist_vis': hist_vis,
            'data_type': ['continuous'] * batch_size,
        }
        
        return batch_dict
    
    def prepare_prompts(self, batch, **kwargs):
        """Prepare model prompts"""
        batch_size = len(batch["instruction"])
        
        hist_nums = [len(his) for his in batch["history"]]
        
        # For continuous navigation, we have 4 actions
        cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        
        prompts = []
        for bn in range(batch_size):
            prompt = self.get_prompt(
                instruction=batch["instruction"][bn],
                hist_num=hist_nums[bn],
                cand_num=int(cand_nums[bn]),
                cls_token=kwargs.get("cls_token"),
            )
            prompts.append(prompt)
        
        return prompts
    
    def get_prompt(self, instruction, hist_num, cand_num, cls_token=None):
        """Build specific prompts"""
        # Task
        prompt = "### Task: Navigate in the continuous environment to complete the given instruction.\n"
        prompt += f"### Instruction: {instruction}\n"
        
        # History
        prompt += f"### Following is the History, which contains the visual information of your previous decisions.\n"
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        
        # Observation
        prompt += f"### Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n"
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        
        # Output Hint
        prompt += f"### Following is the Output Hint, which contains the expected output format.\n"
        prompt += '### Output: {}'.format(cls_token if cls_token else '<cls>')
        
        return prompt
    
    def rollout(self, batch_obs, history, hist_vis, step_id):
        """
        Main forward pass for navigation encoding
        
        Args:
            batch_obs: List of observations
            history: List of history probs for each environment
            hist_vis: List of visual history embeddings for each environment
            step_id: Current step ID
            
        Returns:
            Dictionary containing inputs for navigation model
        """
        # Step 1: Create panorama features
        pano_inputs = self.create_panorama_features(batch_obs)

        # Step 2: Encode panorama features
        pano_outputs = self.model('panorama', pano_inputs)
        pano_embeds = pano_outputs['pano_embeds']  # [batch_size, num_views, hidden_size]
        
        # Step 3: Update graph using panorama embeddings
        self.update_graph(batch_obs, step_id, pano_embeds)
        
        # Step 4: Prepare navigation inputs
        nav_inputs = self.prepare_navigation_inputs(batch_obs, history, hist_vis, pano_embeds)
        
        return nav_inputs
               
    def step(self, batch_obs):
        """
        Execute one navigation step
        
        Args:
            batch_obs: List of observations with format:
                {
                    "instruction": str,
                    "view_feats": numpy.ndarray,  # shape (3, 1024) - left, front, right views
                    "pose": numpy.ndarray,        # shape (4,) [x, y, z, heading]
                }
            
        Returns:
            Dictionary containing navigation outputs
        """
        # Move data to device
        for obs in batch_obs:
            if isinstance(obs['view_feats'], np.ndarray):
                obs['view_feats'] = obs['view_feats'].astype(np.float32)  # (3, 1024)
            if isinstance(obs['pose'], np.ndarray):
                obs['pose'] = obs['pose'].astype(np.float32)
        
        # Prepare navigation inputs visual encoding
        nav_inputs = self.rollout(
            batch_obs, self.history, self.hist_vis, self.step_count
        )
        
        # Move to device
        for key, value in nav_inputs.items():
            if isinstance(value, torch.Tensor):
                nav_inputs[key] = value.to(self.device)
        
        # Prepare prompts
        nav_inputs["prompts"] = self.prepare_prompts(
            nav_inputs,
            cls_token=self.model.lang_model.cls_token[0] if hasattr(self.model, 'lang_model') else '<cls>'
        )
        
        # Execute navigation outputs
        nav_outs = self.model('navigation', nav_inputs)
        
        # Get navigation decisions
        nav_logits = nav_outs['fuse_logits']
        nav_vpids = nav_inputs['gmap_vpids']
        temperature = getattr(self.args, 'temperature', 1.0)
        nav_probs = torch.softmax(nav_logits / temperature, 1)
        a_t, actions = self.get_action(nav_probs)

        # Update history
        batch_size = len(batch_obs)
        for i in range(batch_size):
            self.history[i].append(nav_probs[i].detach())
            if 'fuse_embeds' in nav_outs:
                # Add visual history (use chosen embedding as placeholder)
                self.hist_vis[i].append(nav_outs['fuse_embeds'][i][a_t[i]])
        
        if len(self.history[0]) > self.args.max_memory:
            for i in range(batch_size):
                self.history[i], self.hist_vis[i] = forget_with_entropy(
                    self.history[i], self.hist_vis[i]
                )
        
        self.step_count += 1
        
        # return {
        #     'nav_logits': nav_logits,
        #     'nav_probs': nav_probs,
        #     'nav_vpids': nav_vpids,
        #     'fuse_embeds': nav_outs.get('fuse_embeds'),
        # }
        return actions[0], nav_logits
    
    def get_action(self, nav_probs, method='argmax'):
        """
        Get action from navigation probabilities
        
        Args:
            nav_probs: Navigation probabilities [batch_size, num_actions]
            method: 'argmax' or 'sample'
            
        Returns:
            List of action indices
        """
        if method == 'argmax':
            actions = torch.argmax(nav_probs, dim=1)
        elif method == 'sample':
            actions = torch.multinomial(nav_probs, 1).squeeze(1)
        else:
            raise ValueError(f"Unknown method: {method}")

        str_actions = []
        for action in actions:
            if action == 0:
                str_actions.append('stop')
            elif action == 1:
                str_actions.append('turn_left')
            elif action == 2:
                str_actions.append('move_forward')
            elif action == 3:
                str_actions.append('turn_right')
            else:
                str_actions.append('stop')

        return actions.cpu().tolist(), str_actions