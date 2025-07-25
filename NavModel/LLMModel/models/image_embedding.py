import torch
import torch.nn as nn
from .ops import (
    create_transformer_encoder,
    gen_seq_masks,
    pad_tensors_wgrad
)


class ImageEmbeddings(nn.Module):
    def __init__(self, config, use_obj: bool=False, fuse_obj: bool=False):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

            # if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
        self.fuse_obj = fuse_obj
        if use_obj:
            if self.fuse_obj:
                self.obj_linear = nn.Sequential(
                    nn.Linear(config.obj_feat_size, config.hidden_size),
                    torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
                )
            self.obj_projector = nn.Sequential(
                nn.Linear(config.obj_feat_size, config.output_size),
                torch.nn.LayerNorm(config.output_size, eps=1e-12)
            )
        else:
            self.obj_linear = self.obj_layer_norm = None
            self.fuse_layer = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

        self.mapper = nn.Linear(config.hidden_size, config.output_size)

    def forward_panorama_per_step(self, 
        view_img_fts, 
        view_lens,
        loc_fts=None,
        nav_types=None,
        obj_img_fts=None, 
        obj_lens=None,
        obj_loc_fts=None,
    ):
        ret = {}
        batch_size = view_img_fts.shape[0]
        pano_embeds = self.img_layer_norm(
            self.img_linear(view_img_fts)
        )
        if loc_fts is None:
            loc_fts = torch.zeros(pano_embeds.shape[:2]+(7,), dtype=torch.float).to(pano_embeds.device)
        pano_embeds += self.loc_layer_norm(self.loc_linear(loc_fts))

        if nav_types is None:
            nav_types = torch.ones(pano_embeds.shape[:2], dtype=torch.int).to(pano_embeds.device)
        pano_embeds += self.nav_type_embedding(nav_types)

        pano_embeds = self.layer_norm(pano_embeds)
        pano_embeds = self.dropout(pano_embeds)
        pano_masks = gen_seq_masks(view_lens)
        if self.pano_encoder is not None:

            if self.fuse_obj:
                obj_nav_types = torch.full(obj_img_fts.shape[:2], 2, dtype=torch.int).to(obj_img_fts.device)
                obj_embeds = self.obj_linear(obj_img_fts) + self.loc_layer_norm(self.loc_linear(obj_loc_fts)) + self.nav_type_embedding(obj_nav_types)
                fuse_embeds = []
                for bn in range(batch_size):
                    fuse_embeds.append(
                        torch.cat([
                            pano_embeds[bn, :view_lens[bn]], obj_embeds[bn, :obj_lens[bn]]
                        ], dim=0)
                    )
                fuse_embeds = pad_tensors_wgrad(fuse_embeds)
                fuse_masks = gen_seq_masks(view_lens+obj_lens)
                fuse_embeds = self.pano_encoder(
                    fuse_embeds, src_key_padding_mask=fuse_masks.logical_not()
                )
                pano_embeds = [fuse_embeds[bn, :view_lens[bn]] for bn in range(batch_size)]
                pano_embeds = pad_tensors_wgrad(pano_embeds)

            else:
                pano_embeds = self.pano_encoder(
                    pano_embeds, src_key_padding_mask=pano_masks.logical_not()
                )


        pano_embeds = self.mapper(pano_embeds)
        pano_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0)

        ret.update({
            "pano_embeds": pano_embeds,
            "pano_masks": pano_masks
        })

        # object feature
        if obj_img_fts is not None and obj_img_fts.shape[1] > 0:
            obj_embeds = self.obj_projector(obj_img_fts)
            obj_masks = gen_seq_masks(obj_lens)
            assert obj_embeds.shape[:2] == obj_loc_fts.shape[:2], f'shape of obj_embeds {obj_embeds.shape[:2]} must equal to shape of obj_loc_fts {obj_loc_fts.shape[:2]}'
            ret.update({
                'obj_embeds': obj_embeds,
                'obj_loc_fts': obj_loc_fts,
                'obj_masks': obj_masks
            })

        return ret