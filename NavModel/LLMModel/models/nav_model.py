import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import logging
from .ops import pad_tensors_wgrad, gen_seq_masks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from .image_embedding import ImageEmbeddings
from .modified_lm import ModifiedOPTForCasualLM, ModifiedLlamaForCausalLM, ModifiedQwenForCausalLM, TrieLogitsProcessor
from typing import Dict, List, Any

logging.set_verbosity_error()


def init_vis_config(args, config):
    cfg_name = '/data2/songxinshuai/nav_gen/data/models/bert-large-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.precision = args.precision
    vis_config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    vis_config.max_action_steps = args.max_step
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    # vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.type_vocab_size = 3
    return vis_config


class NavModel(nn.Module):
    def __init__(self, args, logger, model_config):
        super().__init__()
        self.args = args
        config = init_vis_config(args, model_config)
        self.config = config

        # Large Language Model
        if args.resume_from_checkpoint is not None or args.from_scratch:
            logger.info("Initialize the model from config.")
            model_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
            if 'opt' in config.pretrained_model_name_or_path.lower():
                self.lang_model = ModifiedOPTForCasualLM(model_config, config)
            elif 'qwen' in config.pretrained_model_name_or_path.lower():
                self.lang_model = ModifiedQwenForCausalLM(model_config, config)
            else:
                self.lang_model = ModifiedLlamaForCausalLM(model_config, config)
        else:
            self.lang_model = ModifiedOPTForCasualLM.from_pretrained(config.pretrained_model_name_or_path, config) if "opt" in config.pretrained_model_name_or_path \
                else ModifiedLlamaForCausalLM.from_pretrained(config.pretrained_model_name_or_path, config)
        
        self.lang_model.init_tokenizer(config.pretrained_model_name_or_path)

        self.hidden_size = self.lang_model.hidden_size
        self.model_type = self.lang_model.model_type

        # Panorama Encoding
        config.output_size = self.hidden_size
        self.img_embeddings = ImageEmbeddings(config)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.hidden_size)

        # global encoding
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, self.hidden_size)

        # local encoding
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        self.obj_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        # Classfification from candidates
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden_size, 100)
        ).to(self.lang_model.model_type)

        self.instruction = None
        self.history = None
        self.hist_vis = None

        self.drop_env = nn.Dropout(p=args.feat_dropout)

        logger.info("model type: {}".format(self.model_type))


    def forward(self, mode: str, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'panorama':  # batch['view_img_fts'] [B, 36, D=768] --> dropout
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            return self.img_embeddings.forward_panorama_per_step(
                batch['view_img_fts'],
                batch['view_lens'],
                batch['loc_fts'],
                batch['nav_types']
            )

        elif mode == 'navigation':
            return self.forward_navigation(mode, batch, **kwargs)
        else:
            raise NotImplementedError('wrong mode: %s' % mode)
    

    def forward_navigation(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:

        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D=768]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
                gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                                self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                                self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])


        ##### local branch #####
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                        + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        ##### fuse embeds #####
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_vp_embeds[i, j]
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        # fuse_embeds[i, j] += bw_logits
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)

        # cand_embeds = fuse_embeds[cand_masks]  # .to(self.model_type)
        cand_embeds = []
        inv_perms = []
        for bn in range(batch_size):
            # random permute
            cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]
            rand_perm = torch.randperm(cand_embed.shape[0])
            inv_perm = torch.arange(cand_embed.shape[0])
            inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
            inv_perms.append(inv_perm)
            cand_embeds.append(cand_embed[rand_perm]) # remove stop features
        cand_embeds = torch.cat(cand_embeds, dim=0)

        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=cand_embeds,
            hist_vis=hist_vis_input,
        )
        loss, hidden_states = output.loss, output.hidden_states

        fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
            fuse_embeds.device).to(self.model_type)
        
        predictions = self.out_head(hidden_states[text_input['input_ids']==self.lang_model.cls_token_id[0]])

        num_candidates = cand_nums[0].item()
        refine_logits = torch.zeros((fuse_embeds.shape[0], num_candidates)).to(
            fuse_embeds.device).to(self.model_type)
        refine_embeds = torch.zeros((fuse_embeds.shape[0], num_candidates, fuse_embeds.shape[2])).to(
            fuse_embeds.device).to(self.model_type)
        for i in range(batch_size):
            fuse_logits[i][cand_masks[i]] = torch.cat([predictions[i, 0:1],predictions[i, 1:cand_nums[i]][inv_perms[i]]],dim=0)
            refine_logits[i] = fuse_logits[i][cand_masks[i]]
            refine_embeds[i] = fuse_embeds[i][cand_masks[i]]

        return {
            'fuse_embeds': refine_embeds.detach(),
            'fuse_logits': refine_logits,
        }
