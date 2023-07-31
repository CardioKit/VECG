# 1-lead Electrocardiogram Clustering

This project focuses on clustering 1-lead electrocardiogram (ECG) heartbeats using the Total Correlation Variational Autoencoder (TC VAE) technique.
The goal is to detect irregular morphologies in ECG signals, which can serve as indicators of cardiac anomalies like arrhythmia.

## Installation and Setup

To get started with the project, follow these steps:

1. Make sure you have Python version 3.10 installed.
2. Create a virtual environment to isolate the project dependencies:
3. Install the required libraries by running the following command in the project directory:
````
conda create -n ecg python=3.10
conda activate ecg
pip install -r requirements.txt
````

## Data Preparation

The raw ECG data is available in a remote repository and needs to be downloaded and built.
Perform the following steps:

1. Clone the ECG-TFDS repository:
```
git clone https://github.com/CardioKit/ECG-TFDS
```
2. Install the requirements for ECG-TFDS:
```
pip install -r ./ECG-TFDS/requirements.txt
```
3. Change to the ECG-TFDS source directory:
```
cd ./ECG-TFDS/src/zheng
```
4. Build the dataset:
```
tfds build --register_checksums
```

## Running the Code:

Execute the main file to run the code:

```
python main.py 
```
The main file supports several optional parameters:

```
options:
  -h, --help            show this help message and exit
  -r, --path_results    location to save results (default: ../results)
  -d, --dataset         choose tensorflow dataset (default: zheng)
  -l, --label           choose a labelling (default: quality)
  -c, --coefficients    coefficients of KL decomposition (a, b, c)
  -e, --epochs          epochs of train (default: 50)
  -b, --batch_size      batch size for model training (default: 256)
  -s, --seed            seed for reproducibility (default: 42)
  -ld, --latent_dim     dimension of the latent vector space (default: 16)
  -w, --wandb_mode      Disable wandb tracking (default: online)
  -p, --wandb_project   Wandb project name (default: vecg)
```
