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
git clone https://github.com/TypeDefinition/VIBE-Object.git Rendering

pip install tqdm
pip install yacs
pip install h5py
pip install scipy
pip install numpy==1.23
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
pip install chumpy
pip install wget
conda install ffmpeg

mkdir Rendering/data
cd Rendering/data
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip vibe_data.zip
rm vibe_data.zip

# for pose tracking for vibe
git clone -b staf https://github.com/soulslicer/STAF.git openposetrack
cd openposetrack/models

wget -c "https://drive.google.com/uc?export=download&id=1vbwJHQehHXhfYdbvFS-a-6ZBgeFd-AGd" -P "pose/body_25/pose_iter_584000.caffemodel"
wget -c "https://drive.google.com/uc?export=download&id=1gRM0NwOFd-GRRR4p0yCAyZV8w-kTXrNb" -P "pose/coco/pose_iter_440000.caffemodel"
wget -c "https://drive.google.com/uc?export=download&id=1qoDtsnqIgVP-GD-USDMjaU2_Fs60-u0_" -P "pose/mpi/pose_iter_160000.caffemodel"
wget -c "https://drive.google.com/uc?export=download&id=19RJ5NDwyK5W_iNGdUR8jCBCrGtTz26xy" -P "face/pose_iter_116000.caffemodel"
wget -c "https://drive.google.com/uc?export=download&id=1iGU_trITlOu3AXQmGGx1B5KbdimnxJjx" -P "hand/pose_iter_102000.caffemodel"