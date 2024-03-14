# Disentangled Representational Learning of Single Lead Electrocardiogram Signals using Variational Autoencoder

This work focuses on clustering 1-lead electrocardiogram (ECG) heartbeats using beta total correlation variational autoencoder ($\beta$-TCVAE).
The objective is to detect irregular morphologies in ECG signals, which can serve as indicators of cardiac anomalies.

## Installation and Setup

To get started with the project, follow these steps:

1. Make sure you have Python version 3.10 installed.
2. Create a virtual environment
3. Install the required libraries by running the following command in the project directory. Some requirements might need adjustemnt depending on your hardware and OS:
````
conda create -n ecg python=3.10
conda activate ecg
pip install -r requirements.txt
````

## Data Preparation

The raw ECG data is available in a remote repository and needs to be downloaded and built. Therefore, perform the following steps:

1. Clone the ECG-TFDS repository:
```
git clone https://github.com/CardioKit/ECG-TFDS
```
2. Install the requirements for ECG-TFDS:
```
pip install -r ./ECG-TFDS/requirements.txt
```
3. Change to the ECG-TFDS source directory (e.g., Zheng's dataset):
```
cd ./ECG-TFDS/src/zheng
```
4. Build the dataset:
```
tfds build --register_checksums
```

## Running the Code

Execute the main file to run the code:

```
python main.py 
```
The main file requires a configuration file for parameterization:

```
options:
  -h, --help            show this help message and exit
  -p, --path_config     location of the params file (default: ./params.yml)
```

## How to cite?

If you want to either use code or refer to results, please cite the following article: (To be determined)
```
@article{kapsecker2024disentangled,
  title={Disentangled Representational Learning of Single Lead Electrocardiogram Signals using Variational Autoencoder},
  author={Kapsecker, Maximilian and Möller, Matthias C and Jonas, Stephan M},
  journal={TBD},
  volume={TBD},
  number={TBD},
  pages={TBD},
  year={TBD},
  publisher={TBD}
}
```
