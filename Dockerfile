## You should change below region code to the region you used, here sample is use us-west-2
#From 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04 
#From 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04 
#From 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:23.02-py3
RUN pip3 install sagemaker-training

#Remove Cuda 11.8
#RUN apt-get -y purge cuda*
#RUN apt-get -y autoremove
#RUN apt-get -y autoclean
#RUN rm -rf /usr/local/cuda*

#install Cuda 12
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
#RUN cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
#RUN apt-get update
#RUN apt-get -y install cuda

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN update-alternatives --display cuda
RUN update-alternatives --auto cuda
RUN python3 -m pip uninstall -y deepspeed 
#RUN python3 -m pip install pytorch-lightning==1.9.0

#From requirement.txt
RUN python3 -m pip install datasets>=2.8.0
RUN python3 -m pip install sentencepiece>=0.1.97
RUN python3 -m pip install protobuf==3.20.3
RUN python3 -m pip install accelerate>=0.15.0
RUN python3 -m pip install torch>=1.12.0
RUN python3 -m pip install deepspeed>=0.9.0
RUN python3 -m pip install transformers==4.31.0
#RUN python3 -m pip install transformers>=4.31.0,!=4.33.2
RUN python3 -m pip install tensorboard


## Install transfomers version which support LLaMaTokenizer
#RUN python3 -m pip install git+https://github.com/huggingface/transformers.git@68d640f7c368bcaaaecfc678f11908ebbd3d6176

## Make all local GPUs visible
ENV NVIDIA_VISIBLE_DEVICES="all"
