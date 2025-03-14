
### package requirements
pip install matplotlib numpy pandas # general use libraries
pip install opencv-python pillow scikit-image # image-related libraries
pip install ultralytics torch segment_anything tensorflow # machine learning libraries

# make directory for SAM work
mkdir ~/Documents/SAM
cd ~/Documents/SAM
curl -o ./sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
