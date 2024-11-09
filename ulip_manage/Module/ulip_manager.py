import os
import torch

class ULIPManager(object):
    def __init__(self) -> None:
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][ULIPManager::loadModel')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        return True
