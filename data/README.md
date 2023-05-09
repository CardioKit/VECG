# 1-lead Electrocardiogram Clustering

This project uses Variational Autoencoder (VAE) to cluster 1-lead electrocardiogram signals for the detection of arrythmia.
To get started with the project you have to install the required libraries.
Python 3.7 or higher is required to run the software.

````
pip install -r requirements.txt
````

The raw ecg signals are supposed to be stored in the `./data` folder.
The data used in this project and the respective article can be downloaded via

```
mkdir data
cd data
wget https://figshare.com/ndownloader/files/15651326
unzip 15651326
cd ..
```

or from the corresponding website https://figshare.com/collections/ChapmanECG/4560497/2

In case you'd like to add a new data source you have to implement your own data class in the `./dataset.py` file.
The structure and functions have to be of the same format like the objects in place.

You can run the code via

```
python main.py 
```
and might provide the following optionals

```
  -h, --help            show this help message and exit
  --preloaded N         use preloaded data in ./temp folder
  --epochs N            number of epochs to train (default: 5000)
  --batch_size N        input batch size for training (default: 512)
  --seed S              random seed (default: 42)
  --latent_dim N        dimension of the latent vector space (default: 32)
  --intermediate_dim N  dimension of the intermediate layer (default: 128)
  --final_length N      length of preprocessed signal (default: 1000)
  --seconds N           seconds the preprocessed signal represents (default: 10)
  --model N             load stored model
  --beta N              beta parameter of beta-VAE (default: 1.0)
```
