# Material Density of States Prediction via Multi-Modal Transformer
<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www.ml4materials.com/" alt="Workshop Conference">
        <img src="https://img.shields.io/badge/ICLR 2023 ML4Materials-brightgreen" /></a>
</p>
The offical source code for Material Density of States Prediction via Multi-Modal Transformer paper, accepted at ICLR 2023 Workshop on Machine Learning for Materials (ML4Materials)

### Overview
The density of states (DOS) is a spectral property of materials, which provides
fundamental insights on various characteristics of materials. In this paper, we propose 
to predict the density of states (DOS) by reflecting the nature of DOS: DOS
determines the general distribution of states as a function of energy. 
Specifically, we integrate the heterogeneous information obtained from the crystal structure and
the energies via multi-modal transformer, thereby modeling the complex relation-
ships between the atoms in the crystal structure, and various energy levels. Exten-
sive experiments on two types of DOS, i.e., phonon DOS and electron DOS, with
various real-world scenarios demonstrate the superiority of DOSTransformer.  
![FIG_1](https://user-images.githubusercontent.com/62690984/223295679-55c6c32b-629d-4dae-b8a9-352992a8177e.png)  

### Phonon DOS Prediction
#### Dataset
You can dowload phonon dataset in this [repository](https://github.com/ninarina12/phononDoS_tutorial)  

#### Run model
Run `main_phDOS.py` for phonon DOS Prediction after downloading phonon dataset into `data/processed`

### Electron DOS Prediction
#### Dataset
We build Electron DOS dataset consists of the materials and its electron DOS information which are collected from [Materials Proejct](https://materialsproject.org/)  
We converted raw files to `pkl` and made electronic DOS dataset by `mat2graph.py`  

#### Run model
Run `main_eDOS.py` for electron DOS Prediction after building electron dataset.   

### Models
#### embedder eDOS
`DOSTransformer.py`: Our proposed model / `graphnetwork.py`: GraphNetwork using Energy Embedding   
`graphnetwork2.py`: GraphNetwork not using Energy Embedding /  `mlp.py`: Mlp using Energy Embedding  
`mlp2.py`: Mlp not using Energy Embedding  
#### embedder phDOS
`DOSTransformer_phonon.py`: Our proposed model / `graphnetwork_phonon.py`: GraphNetwork using Energy Embedding   
`graphnetwork2_phonon.py`: GraphNetwork not using Energy Embedding /  `mlp_phonon.py`: Mlp using Energy Embedding  
`mlp2_phonon.py`: Mlp not using Energy Embedding  

### Hyperparameters  

`--layers:` Number of GNN layers in DOSTransformer model  

`--transformer:` Number pf Transformer layer in DOSTransformer   

`--embedder:` Selecting embedder   

`--hidden:` Size of hidden dim  

`--epochs:`  Number of epochs for training the model

`--lr:` Learning rate for training the model  

`--dataset:` Selecting dataset for eDOS prediction (Random split, Crystal OOD, Element OOD, default dataset is Random split)

`--es:` Early Stopping Criteria  

`--eval:` Evaluation Step  
