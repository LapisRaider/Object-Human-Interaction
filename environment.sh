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
