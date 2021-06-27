FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y curl apt-utils
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install -y vim make gcc wget tar unzip git
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6


RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN add-apt-repository -y ppa:deadsnakes/ppa

# RUN python3 -m pip install --upgrade pip && python3 -m pip install -U tensorflow==2.4
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# install miniconda
ENV CONDA_DIR=/root/miniconda3
ENV PATH=${CONDA_DIR}/bin:${PATH}
ARG PATH=${CONDA_DIR}/bin:${PATH}
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /opt

COPY Dockerfile /opt
COPY environment.yml /opt

# install conda env
RUN conda config --set ssl_verify no
RUN conda env create -f environment.yml -n env_bud \
    && conda init bash \
    && echo "conda activate env_bud" >> /root/.bashrc

