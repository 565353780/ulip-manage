import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from PIL.ImageFile import ImageFile
from typing import Union
from collections import OrderedDict

from utils.tokenizer import SimpleTokenizer

from ulip_manage.Model.ulip2_with_openclip import ULIP2WithOpenCLIP

class Detector(object):
    def __init__(self, args, model_file_path: Union[str, None] = None,
                 open_clip_model_file_path: Union[str, None] = None,
                 device: str = 'cuda:0') -> None:
        self.model = ULIP2WithOpenCLIP(args, open_clip_model_file_path, device)
        self.device = device
        self.tokenizer = SimpleTokenizer()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        ckpt = torch.load(model_file_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        self.model.to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print("=> loaded pretrained checkpoint '{}'".format(model_file_path))
        return True

    @torch.no_grad()
    def encodeText(self, texts: Union[list, str]) -> torch.Tensor:
        texts = self.tokenizer(texts).to(self.device, non_blocking=True)
        if len(texts.shape) < 2:
            texts = texts.unsqueeze(0)
        text_embeddings = self.model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.mean(dim=0)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    @torch.no_grad()
    def encodeImage(self, image: Union[torch.Tensor, ImageFile]) -> torch.Tensor:
        image_embedding = self.model.encode_image(image)
        return image_embedding

    @torch.no_grad()
    def encodeImageFile(self, image_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::encodeImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image = Image.open(image_file_path)
        image_embedding = self.encodeImage(image)[0]
        return image_embedding

    @torch.no_grad()
    def encodePointCloud(self, points: torch.Tensor) -> torch.Tensor:
        pc_features = self.model.encode_pc(points)[0]
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        return pc_features

    @torch.no_grad()
    def encodePointCloudFile(self, points_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(points_file_path):
            print('[ERROR][Detector::encodePointCloudFile]')
            print('\t points file not exist!')
            print('\t points_file_path:', points_file_path)
            return None

        if points_file_path.endswith('.npy'):
            points = np.load(points_file_path)
        elif points_file_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(points_file_path)
            points = np.asarray(pcd.points)

        points = torch.from_numpy(points).type(torch.float32).to(self.device).unsqueeze(0)
        return self.encodePointCloud(points)
