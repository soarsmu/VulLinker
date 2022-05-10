FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y git libsndfile-dev && apt-get clean
RUN git clone https://github.com/nvidia/apex /apex && \
    cd /apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN pip install -r requirements.txt