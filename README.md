# ShadowCast: Controllable Graph Generation

SHADOWCAST is a generative model based on a conditional generative adversarial network, capable of controlling graph generation while retaining the original graph's intrinsic properties. 

![shadowcast_architecture](/image/ShadowCast_architecture.png)


## Cite

Please cite our paper if you find this code useful for your own work:

```
@article{tann2020shadowcast,
	title={SHADOWCAST: Controllable graph generation},
	author={Tann, Wesley Joon-Wie and Chang, Ee-Chien and Hooi, Bryan},
	journal={arXiv preprint arXiv:2006.03774},
	year={2020}
}
```

Many times, data of various situations are not available in observed real-world networks. For example, email communications in an organization between various departments. Due to limited data, previously observed network information may be missing scenarios of intra-department email surge within either the Human Resources or Accounting departments. 

![example](/image/Shadowcast_gen-email.png)

Given an observed graph and some user-specified Markov model parameters, SHADOWCAST controls the conditions to generate desired graphs.

## Prerequisites

Installing package requirements:
```
pip install -r requirements.txt
```

## Data

Our datasets are in the [data](/data/) folder.
1. EUcore-top
2. Enron 
3. Cora-ML



## Running the experiments
All the experiments of reported results are in the notebooks listed below:
* eggen-EUcoretop.ipynb
* eggen-enron.ipynb
* eggen-coraml.ipynb
* control-enron.ipynb
