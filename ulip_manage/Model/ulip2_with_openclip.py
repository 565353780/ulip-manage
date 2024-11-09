import torch
import open_clip
import numpy as np
from torch import nn
from typing import Union
from PIL.ImageFile import ImageFile

from ulip_manage.Method.config import cfg_from_yaml_file
from ulip_manage.Model.PointBERT.point_transformer import PointTransformer


class ULIP2WithOpenCLIP(nn.Module):
    def __init__(self, args, open_clip_model_file_path: Union[str, None] = None):
        super().__init__()
        pc_feat_dims = 768

        print("Get openclip model:")
        if open_clip_model_file_path is None:
            open_clip_model_file_path = 'laion2b_s39b_b160k'
        self.open_clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=open_clip_model_file_path)
        self.open_clip_model.eval()
        print("Finished loading the openclip model.")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        config_addr = './ulip_manage/Config/PointTransformer_8192point.yaml'
        config = cfg_from_yaml_file(config_addr)
        self.point_encoder = PointTransformer(config.model, args=args)

        self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

        self.pc_projection = nn.Parameter(torch.empty(pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=1280 ** -0.5)

    def encode_image(self, image: Union[torch.Tensor, ImageFile]) -> torch.Tensor:
        if isinstance(image, ImageFile):
            image = self.preprocess_val(image).unsqueeze(0).to(self.device)
        x = self.open_clip_model.encode_image(image)
        return x

    def encode_text(self, text: Union[list, str, torch.Tensor]):
        x = self.open_clip_model.encode_text(text)
        return x

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}
