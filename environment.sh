ENV_NAME="Obj_Human_Interaction"

conda create --name ${ENV_NAME} python=3.8
conda activate ${ENV_NAME}

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c nvidia
# pip install --upgrade setuptools
# pip install numpy==1.25.2
# pip install opencv-python==4.8.0.76

# for config file
conda install pyyaml

# yolov8 for object classification
pip install ultralytics==8.1.0

#object tracking
pip install tensorflow
git clone https://github.com/LapisRaider/deep_sort.git

# for vibe
git clone https://github.com/TypeDefinition/VIBE-Object.git

pip install tqdm
pip install yacs
pip install h5py
pip install scipy
pip install numpy
pip install smplx
pip install gdown
pip install PyYAML
pip install joblib
pip install pillow
pip install trimesh
pip install pyrender
pip install progress
pip install filterpy

pip install matplotlib
pip install tensorflow
pip install tensorboard
pip install scikit-image
pip install scikit-video
pip install opencv-python
pip install llvmlite
pip install pytube
conda install ffmpeg

mkdir VIBE-Object/data
cd VIBE-Object/data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip vibe_data.zip
rm vibe_data.zip