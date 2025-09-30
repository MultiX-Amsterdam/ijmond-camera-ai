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

# <a name="install-nvidia"></a>Install NVIDIA drivers, CUDA, and PyTorch (administrator only)
> WARNING: this section is only for system administrators, not developers.

This installation guide assusmes that you are using the Ubuntu 22.04 server version (not desktop version) operating system.

## Disable the nouveau driver

Run the following on open the file (assuming that you use `vim` as the text editor):
```sh
sudo vim /etc/modprobe.d/blacklist.conf
```
Then, add the following to the file to blacklist nouveau driver:
```sh
# Blacklist nouveau driver (for NVIDIA driver installation)
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```
Next, regenerate the initial ram file system of the kernel and reboot the computer:
```sh
sudo update-initramfs -u
sudo reboot now
```
Then, check if nouveau is disabled correctly using the following command. You should not see any outputs from the terminal.
```sh
lsmod | grep -i nouveau
```

## Remove old NVIDIA driver
We need to remove old NVIDIA drivers before installing the new one. For drivers that are installed using the files that are downloaded from NVIDIA website, run the following:
```sh
# For drivers that are installed from NVIDIA website file, remove the driver using the following command:
sudo nvidia-uninstall
```
For drivers that are installed using `sudo apt-get`, run the folloing:
```sh
# For drivers that are installed using sudo apt-get, , remove the driver using the following commands:
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get autoremove
```
After that, run the following to check if the drivers are removed. You should not see anything from the terminal.
```sh
lsmod | grep nvidia
```
Also, run the following, and it should tell you some kind of error message that it does not exist:
```sh
nvidia-smi
```

## Install new NVIDIA driver
Next, install NVIDIA driver. Run the following to identify the GPU:
```sh
lspci | grep -i nvidia
```
Then, run the following to check the drivers:
```sh
ubuntu-drivers devices
```
Identify the recommended version (with the "distro non-free recommended" text) and then check the [PyTorch website](https://pytorch.org/) about the supported CUDA versions.
After that, check this [NVIDIA CUDA page] to see if the driver supports the intended CUDA version. If yes, run the following command to install the driver.
Otherwise, pick another NVIDIA driver version or and older PyTorch version that works on older CUDA.
```sh
sudo apt-get install nvidia-driver-570
```
Replace `570` with the version that you want to install. After that, reboot the system and check if the driver is working:
```sh
sudo reboot now
nvidia-smi
```
You should see something on the terminal, indicating the status of the GPU.
Also, running the following should now show some installed drivers:
```sh
lsmod | grep nvidia
```

## Install PyTorch
Then, install `pytorch` from the [official website](https://pytorch.org/).
Make sure to choose the one with the CUDA version that works with the NVIDIA driver that you just installed.
The `pip` installation version comes with CUDA, so you do not need to install CUDA separately.
The command should look similar with the one below:
```sh
conda activate ijmond-camera-ai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
After that, use the following to verify if PyTorch can use CUDA.
```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
You should see something like below on the terminal:
```sh
True
NVIDIA GeForce RTX 3090
```
