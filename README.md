# Code implementation for : [Graph Neural Network-based Multivariate Fault Detection for Wind Turbines Using SCADA Data]


# Installation
### Requirements
* Python >= 3.6
* cuda == 11.3
* [Pytorch==1.11.0](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/openSCADA/

data
 |-openSCADA
 | |-features.txt # the process-knowledge graph given by coo format
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-features.txt # the process-knowledge graph given by coo format
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* Raw SCADA data of four turbine can be found in folder openSCADA_data/raw_data
* Processd and used SCADA data in the paper can be found in folder openSCADA_data/processed


# Citation
If you find this repo or our work useful for your research, please consider citing the paper

