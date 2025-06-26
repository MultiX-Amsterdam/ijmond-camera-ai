# ijmond-camera-ai

This repository hosts code for the AI model for industrial smoke segmentation using data from the [IJmond camera monitoring system](https://github.com/MultiX-Amsterdam/ijmond-camera-monitor).

## 1. Folder: samples_for_labelling
This folder includes all the necessary code for sampling images. These images will be labeled in the next stage.

## 2. Folder: oracle_evaluator
This folder contains the files for the annotator tool that was used to label the sampled images from the previous step.

## 3. Folder: bvm_training
This folder contains the code for the training of bvm model for both versions; original and semi-supervised.

## 4. Folder: bvm_docker
This folder contains the files used to build the Docker for running inference on the BVM model. Please refer to the readme file in the folder for more details.

# <a name="install-conda"></a>Setup the conda environment (administrator only)
> WARNING: this section is only for system administrators, not developers.

Install conda for all users.
This assumes that Ubuntu is installed.
A detailed documentation is [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
First visit [here](https://conda.io/miniconda.html) to obtain the downloading path.
The following script install conda for all users:
```sh
# For Ubuntu
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
sudo sh Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -b -p /opt/miniconda3
echo '' | sudo tee -a /etc/bash.bashrc
echo '# For miniconda3' | sudo tee -a /etc/bash.bashrc
echo 'export PATH="/opt/miniconda3/bin:$PATH"' | sudo tee -a /etc/bash.bashrc
echo '. /opt/miniconda3/etc/profile.d/conda.sh' | sudo tee -a /etc/bash.bashrc
source /etc/bash.bashrc
```
For Mac OS, I recommend installing conda by using [Homebrew](https://brew.sh/).
```sh
# For Mac OS
brew install --cask miniconda
echo 'export PATH="/usr/local/Caskroom/miniconda/base/bin:$PATH"' >> ~/.zshrc
echo '. /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh' >> ~/.zshrc
source ~/.bash_profile
```
Create conda environment and install packages.
It is important to install pip first inside the newly created conda environment.
```sh
conda create -n ijmond-camera-ai
conda activate ijmond-camera-ai
conda install python=3.13
conda install pip
which pip # make sure this is the pip inside the conda environment
sh ijmond-camera-monitor/back-end/install_packages.sh
```
If the environment already exists and you want to remove it before installing packages, use the following:
```sh
conda deactivate
conda env remove -n ijmond-camera-ai
```