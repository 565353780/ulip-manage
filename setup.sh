cd ..
git clone https://github.com/fishbotics/pointnet2_ops.git

cd pointnet2_ops
python setup.py install --user

pip install -U easydict timm ftfy regex pyyaml_env_tag \
  h5py open3d wandb termcolor lmdb open-clip-torch

pip install -U torch torchvision torchaudio
