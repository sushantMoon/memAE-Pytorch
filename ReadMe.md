# Original Work

[Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](https://arxiv.org/abs/1904.02639)

Author's Code : [Github](https://github.com/donggong1/memae-anomaly-detection)

# Why code again

Author's code feels incomplete and buggy.
It has the model implemented for video input but training, loss and data loader functions are missing. 

I needed to use memAE for vector inputs hence this script with dummy data. 

Hope it helps others as well.

# Information on the Repo

[memAE](./memAE) : main folder under which all the scripts are present.
   - [data](./memAE/data) : data has the scripts for data ingestion, dataloader specifically. Write your own pre-processing scripts here if needed.
   - [models](./memAE/models) : has the scripts related to model architecture.
   - [utils](./memAE/utils) : has the loss functions. Add your own utility functions here.

[run.py](./run.py) : script which has training code, and validation code. It should be only used as a reference point. *Check for bugs, use at your own risk*. 

# Major Requirements

* PyTorch 1.6
* Numpy 1.18.5
* Pandas 1.0.5


