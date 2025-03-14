# LeafSLA
Pipeline for automated measurements of leaf images

##### Code Requirements:
 - python≥3.12.2 (& miniconda)
 - python libraries listed in requirements.sh
 - SAM vit.b model version (though others are adaptable into the pipeline)–this will auto-download during the execution of requirements.sh

##### Image Requirements:
 - simple background with leaf in focus and well-lit
 - a red square for size comparison. can vary in size but an integer-based side length is recommended (e.g., 1cm*1cm)

##### To run out of the box:
on a system without python/miniconda/coding IDE installed:
 - visit Anaconda and download the appropriate Miniconda installer: https://www.anaconda.com/download/success
    - follow the instructions for miniconda setup 
 - Visual Studio Code is recommended for processing images, but any IDE that works with .ipynb will work
    - VS Code download link: https://code.visualstudio.com/download 
    - in VS Code, add the python and Jupyter extensions in VS Code (on side bar, click the 4-square icon and search for these)
 - in a terminal window (in VS Code click Terminal on the top menu bar -> new terminal) check python is up and working by entering: python
    - should open python in-line and allow you to script (look for >>>)
    - enter: quit() <- allows you to exit python prompter
    - continue following instructions below
 
 if you already have python/miniconda/coding IDE installed, begin here:
 - in terminal, navigate to the LeafSLA folder,then enter: zsh requirements.sh 
    - to navigate, use cd ~/path/to/LeafSLA
    - this script will download SAM and install required packages for leaf_SLA.py and pipeline.ipynb
    - this script downloads SAM to a new directory: ~/Documents/SAM
 - you should now be able to open and run the pipeline.ipynb notebook
    - multiple variables are supplied user-side, though set with pre-definitions in the notebook. see notebook for further details
