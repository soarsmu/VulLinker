# VulLinker: Automated Affected Library Detection from Vulnerability Report Data


## Preparation
Users can choose whether to prepare VulLinker from scratch or use the docker image. For ease of use, users can choose to use docker image as the required libraries are already installed to ensure that VulLinker can run smoothly.

### Prepare from Scratch
- To start using VulLinker, we need to clone this project
- Then, we can install the required libraries in the requirements.txt. For an easier virtual environment, we can use conda env.

```conda install --file requirements.txt```

Keep in mind when installing the requirements, it uses the NVidia Apex (https://github.com/NVIDIA/apex). If apex is not installed correctly when using requirements.txt, we need to manually install it. The command used for Windows:

```pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext```

While for linux:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- To prepare the dataset for the training and testing, we can use the script in “<working_directory>/Utilities>/data_preparation.py”. We can actually skip this step as the processed dataset is already included in the repository. To run the data preparation, we can use the following command:

```cd <working_directory>/Utilities>/
   python data_preparation.py
```

- Then, if we choose to run the dataset_preparation, we need to move the processed dataset into “<working_directory>/LightXML/data” folder.
- To run the training and evaluation script, we can use the run.sh script in the “<working_directory>/LightXML/” folder.

```./run.sh cve_data```


### Use Docker Image
For ease of use, we provide a docker image that can be accessed in:
https://hub.docker.com/repository/docker/ratnadira/vullinker
We can run and pull the docker image using below command:

```docker run --name=<docker_name> --gpus '"device=0,1"' --shm-size 32G -it -p 8000:8000 ratnadira/vullinker```

Noted that we can change the gpus parameter based on the spec that what we have.


## Run VulLinker
To run VulLinker after the preparation stage is finished, please make sure that we are on the right working directory. We can move to the directory using the command:

```cd <working_directory>/Web/```

In this example, the working directory for the website is at “/workspace/tool/VulnerabilityDetection/Web/”. After we are in the right directory, we can migrate and run the server using the below commands:

```
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```




