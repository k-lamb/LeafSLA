
### package requirements
pip install matplotlib numpy pandas # general use libraries. versions are: matplotlib=3.10.1 numpy=1.26.4 pandas=2.2.3
pip install opencv-python pillow scikit-image # image-related libraries. versions are: opencv=4.11.0.86 pillow=11.1.0 scikit-image=0.25.2
pip install torch segment_anything tensorflow # machine learning libraries. versions are: segment_anything=1.0 tensorflow=2.19.0

# make directory for SAM work
mkdir ~/Documents/SAM
cd ~/Documents/SAM
curl -o ./sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
