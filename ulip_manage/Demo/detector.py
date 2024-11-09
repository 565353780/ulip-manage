from ulip_manage.Config.config import getConfig
from ulip_manage.Module.detector import Detector

def demo():
    args = getConfig()
    model_file_path = '/home/chli/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
    device = 'cpu'

    detector = Detector(args, model_file_path, open_clip_model_file_path, device)

    return True
