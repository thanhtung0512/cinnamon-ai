# INRODUCTION
This project endeavors to create a robust Optical Character Recognition (OCR) system tailored specifically for recognizing Vietnamese handwritten text. Its primary objective is to accurately detect and interpret text inscribed in a single line. Powered by deep learning models adept at handling the intricacies of handwritten text, the system is trained using a blend of real and synthetic data to ensure superior accuracy and versatility.

The project is structured around the following key phases:

- Training Phase: Setting up the environment, data preparation, model training using synthetic and real datasets, and fine-tuning for optimal performance.
- Inference Phase: Employing the trained models to make predictions on new, unseen data.
- Web Application: Developing an intuitive web interface to facilitate image uploads and OCR predictions.

This documentation provides comprehensive guidance on installing, preparing data, training models, and utilizing the OCR system and its associated web application. Additionally, it includes a technical overview detailing the key methodologies and components employed in the project.


# TECHNICAL OVERVIEW
I started with the available VietOCR source code on GitHub. To ensure the model size meets the requirements (smaller than 50MB), I replaced the existing backbone with lighter backbones (available in the timm library). Additionally, I used float16 precision to reduce the model size. Specifically, I used two models: EfficientNetV2_B1 (24.5MB) and EfficientNetV2_B2 (22.8MB). The total size of these two models is 47.3MB, which is within the allowed limit.

To enhance accuracy, I created additional synthetic datasets:
- Synthetic Dataset 1: Augmented version of the competition data.
- Synthetic Dataset 2: Randomly generated addresses using province and district names, converted into image data using PIL.
- Synthetic Dataset 3: Generated image data from existing poems.
# TRAINING PHASE
## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 11.2
- Python 3.7.5
- Training PC: 1x RTX3090 (or any GPU with at least 24Gb VRAM), 32GB RAM.
- python packages are detailed separately in requirements.txt
```
$ conda create -n envs python=3.7.5
$ conda activate envs
$ pip install -r requirements-training.txt
$ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## 2.DATA
* Kalapa dataset.  
* Vietnamese address dataset `https://github.com/thien0291/vietnam_dataset`  
* Vietnamese poems corpus: `https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus`  
* Synthetic data. Download my generated data here (https://drive.google.com/file/d/1FfjVZqNRZGExZjZmqzWk_L3QDQS-20Bl/view?usp=sharing)  
* Or following the commands below to re-generating it. (make sure poems_dataset.csv and vietnam_dataset are inside synthetic_data/)    
```
$ cd synthetic_data  
$ python gendata_address.py    
$ python gendata_aug.py  
$ python gendata_poems.py  
$ cd ..  
$ python prepare_ext_data.py  
``` 

* Folder structure before executing training  
├── training_data   
│ ├── images    
│ ├── annotations    
├── synthetic_data   
│ ├── address    
│ ├── aug    
│ ├── poems  
│ ├── ...  
├── configs   
│ ├── b2_256_ptr_f5.py   
│ ├── b1_384_ptr_f5.py   
│ ├── ...   
├── train.py  
├── prepare_ext_data.py  
├── train_ext3.csv  
├── train_folds.csv  
├── ...  

## 3.TRAINING
* Pretrained models on synthetic data.  
```
$ python train.py -C b2_256_ptr_f5  
$ python train.py -C b1_384_ptr_f5  
```

* Fine-tune models on real data.  
```
$ python train.py -C b1_384_f5  
$ python train.py -C b2_256_f5  
```

## 4.INFERENCE

* Refer to submitted notebook


# HOW TO RUN THIS WEB-APP
## 1. INSTALLATION
1. **Download OCR Test Dataset**:
   - Download the dataset from [this link](https://drive.google.com/drive/folders/1s3mGm31XuI5v8Q2__-Y5m_9vZZQXtqwI?usp=sharing) and save it to the current folder.

2. **Update Script Paths**:
   - Open `frontend/script.js`.
   - Change the `folderPaths` values to the absolute paths of the downloaded dataset.

3. **Install Required Packages**:
   - Run the following command to install all required packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Start the Flask API Server**:
   - Use the following command to start the server:
     ```bash
     python main.py
     ```

5. **Open the Website**:
   - Open `frontend/index.html` in your web browser to use the website.

## 2. RUN USING DOCKER
* Build the Docker image:
   ```bash
   docker build -t kalapa-app .
   ```
*  Run the Docker image:
   ```bash
   docker run --network=host -it --rm -p 5000:5000 kalapa-app
   ```
* Open `frontend/index.html` in your web browser to use the website.


## 2. Usage

* This OCR website detects Vietnamese handwritten text written in one line. When you open the website, you will see two buttons: "Get Images" and "Predict".

* **Get Images**:
  - Clicking this button will randomly select 3 images from the test images folder and display them on the website.

* **Predict**:
  - After displaying the images, clicking this button will send the selected image to the server. The server will then process the image using the OCR model and return the predicted text.

## 3. Demo video



 