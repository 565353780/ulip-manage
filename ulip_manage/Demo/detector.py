from ulip_manage.Config.config import getConfig
from ulip_manage.Module.detector import Detector

def demo():
    args = getConfig()
    model_file_path = '/home/chli/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
    device = 'cpu'

    detector = Detector(args, model_file_path, open_clip_model_file_path, device)

    pcd_file_path = '/home/chli/chLi/Dataset/SampledPcd_Manifold/ShapeNet/03001627/46bd3baefe788d166c05d60b45815.npy'
    points_feature = detector.encodePointCloudFile(pcd_file_path)
    print('points_feature:')
    print(points_feature.shape)

    texts = ['an chair', 'a beafutiful chair']
    text_feature = detector.encodeText(texts)
    print('text_feature:')
    print(text_feature.shape)

    image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/03001627/46bd3baefe788d166c05d60b45815/y_5_x_3.png'
    image_feature = detector.encodeImageFile(image_file_path)
    print('image_feature:')
    print(image_feature.shape)

    return True
