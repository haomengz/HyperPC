FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install scipy && \
    pip install --ignore-installed PyYAML && \
    pip install open3d

## Install PyG
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install h5py && \
    pip install tensorboard==2.3.0 && \
    pip install protobuf==3.20.*

# Install graspnetAPI
RUN git clone https://github.com/graspnet/graspnetAPI.git
RUN cd graspnetAPI && pip install .

# Fixed RuntimeError: Ninja is required to load C++ extension 
RUN apt-get install -y wget unzip
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

# compile pointnet2_utils for snowflakenet
RUN git clone https://github.com/AllenXiangX/SnowflakeNet.git
RUN cd SnowflakeNet/models/pointnet2_ops_lib && python setup.py install

# Add user to solve 'I have no name' warning
ARG UNAME=DockerUser
ARG UID=1001
ARG GID=1001
RUN groupadd -g ${GID} $UNAME
RUN useradd -l -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME}

USER ${UNAME}
WORKDIR /home/$UNAME
