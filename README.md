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



## How to Use
Each folder within this repository contains the implementation of different XML approaches that can be applied to the CVE data to identify the library. The Utility folder contains utility functions that may be useful to ease the process of using this repository.

### QuickLink:
- [Utility Folder](#Utility-Folder)
- [FastXML](#FastXML)
- [Omikuji](#Omikuji)
- [LightXML](#LightXML)

## Utility Folder
**data_preparation.py**: contains functions that can be used to prepare the dataset for different XML algorithm. The inputs for most of these functions are mainly the **pre-splitted dataset available in the dataset/splitted folder**
Please make sure that the folder exist in the Utility/dataset/ directory before using the functions in data_preparation.py. An alternative is to modify the functions to include folder creation functionality.

**dataset folder**: contains the dataset of the CVE data, both in the splitted form and in the original csv form. All the results from the data_preparation.py functions will be available here

## LightXML
### Environment setup
- Please refer to https://github.com/kongds/LightXML/
- For easier virtual environment, I recommend to use conda env
- Then, install the requirements listed in requirements.txt
- Keep in mind when installing the requirements, it use the NVidia Apex (https://github.com/NVIDIA/apex). The one listed in requirements.txt is often linked with the wrong library

### Training and Evaluation
- To run the training and evaluation script, use the run.sh script
- ./run.sh cve_data
- Refer to line 76--86 of the run.sh script

## FastXML
### Environment setup
- I use Python 3.6 virtual environment for FastXML
- After creating the virtual environment, install the libraries listed in the requirements.txt

### Data preparation
- For the FastXML, I use the json file structure indicating the train and test data as suggested in the FastXML repo readme
- Data preparation utility function is available in Utilities/data_preparation.py **prepare_fastxml_dataset()** function. 
- This function make use of the splitted_train_x.npy, splitted_test_x.npy (the pre-splitted numpy dataset), and the cve_labels_merged_cleaned.csv (the csv file containing all the entries)
- To make the dataset consistent, it would be good to use the **dataset_train.csv** and **dataset_test.csv** available in the utilities/dataset/splitted/splitted_dataset_csv.zip and change the **merged column** to the text that we want as the feature.
- Then, to convert these two csv files into the numpy array, we can use the **split_dataset()** function in the data_preparation file.
- After we have created the **train.json** and **test.json** for FastXML, copy the two files into FastXML/dataset folder

### Training Process
- To start the training process, run the FastXML/baseline.py. We need to define the run parameters. For starter, we can use the following parameter which produce similar result to Veracode's implementation:

    

>     model/model_name.model dataset/path_to_train.json --verbose train --iters 200
>     --gamma 30 --trees 64 --min-label-count 1 --blend-factor 0.5  --re_split 0 --leaf-probs

- After the training process is completed, the model will be created in the FastXML/model folder
- Then, we run the FastXML/baseline.py again for the model testing with the following run parameter:

> `model/path_to_model_folder dataset/path_to_test.json inference --score`

- Running the above test command will produce FastXML/inference_result.json which contains the inference result of the model. 
- To calculate the precision, recall, and F1 metrics, run the FastXML/util.py, which will calculate the metric from the inference_result.json file.

## Omikuji 
Omikuji is the name of the library that provides the implementation of both Bonsai and Parabel. It is fairly straightforward to setup omikuji as it is readily available in the form of a library.

### Data Preparation
- Omikuji takes as input training and test data in the form of svmlight file of the Tf Idf features of the data
- Data preparation utility function is available in Utilities/data_preparation.py prepare_omikuji_dataset function
- This utility function make use of the pre-splitted numpy array dataset

### Environment Setup
- Install the Python binding of Omikuji as specified in its repository README (https://github.com/tomtung/omikuji/).
   > pip install omikuji
- For the omikuji library I use Python 3.8 environment and omikuji version 0.3.2.
- If there is an error with the omikuji installation, please consider manually installing Omikuji from the repository.
- The above Python binding of Omikuji installation is used for the model prediction purpose.
- Meanwhile, for the model training using Omikuji, I use the Rust implementation of Omikuji that is available in Cargo (Refer to Build & Install section of Omikuji repository)

### Training Process
- After Omikuji is successfully installed from Cargo, we can use the following command to train a model:
**Parabel Model**
> `omikuji train --model_path model_output_path --min_branch_size 2  --n_trees 3 path_to_dataset`

**Bonsai Model**

> `omikuji train --cluster.unbalanced --model_path model_output_path 
> --n_trees 3 dataset/train.txt`

-  Then, we can use the created models to predict the test data by running the Omikuji/omikuji_predict.py with model_path and test_data_path run parameters

## LightXML



## Deployment - Deploy LightXML models

Deploy the model using Django
```
# don't forget to use absolute path
docker run --rm --name=xml --gpus '"device=0,1"' --shm-size 32G -it --mount type=bind,src=<absolute path to the folder>,dst=<folder name>/ -p 8000:8000 username/xml
cd <folder name>/Web/
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

we can try to open 0.0.0.0:8000 on wer browser to check the prediction. Alternatively, we can check the prediction from command line by `curl http://0.0.0.0:8000/predict/?input_text=imagemagick+attackers+service+segmentation`


