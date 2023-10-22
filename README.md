# Density of States Prediction of Crystalline Materials via Prompt-based Multi-Modal Transformer

The offical source code for Density of States Prediction of Crystalline Materials via Prompt-based Multi-Modal Transformer
 
### Overview
The density of states (DOS) is a spectral property of crystalline materials, which provides fundamental insights into various characteristics of the materials. While previous works mainly focus on obtaining high-quality representations of crystalline materials for DOS prediction, we focus on predicting the DOS from the obtained representations by reflecting the nature of DOS: DOS determines the general distribution of states as a function of energy. That is, DOS is not solely determined by the crystalline material but also by the energy levels, which has been neglected in previous works. In this paper, we propose to integrate heterogeneous information obtained from the crystalline materials and the energies via a multimodal transformer, thereby modeling the complex relationships between the atoms in the crystalline materials and various energy levels for DOS prediction. Moreover, we propose to utilize prompts to guide the model to learn the crystal structural system-specific interactions between crystalline materials and energies. Extensive experiments on two types of DOS, i.e., Phonon DOS and Electron DOS, with various real-world scenarios demonstrate the superiority of DOSTransformer.

### Phonon DOS Prediction
#### Dataset
You can dowload phonon dataset in this [repository](https://github.com/ninarina12/phononDoS_tutorial)  

#### Run model
Run `main_phDOS.py` for phonon DOS Prediction after downloading phonon DOS dataset into `data/processed`

### Electron DOS Prediction
#### Dataset
We build Electron DOS dataset consists of the materials and its electron DOS information which are collected from [Materials Proejct](https://materialsproject.org/)  
We converted raw files to `pkl` and made electronic DOS dataset by `mat2graph.py`  

#### Run model
Run `main_eDOS.py` for electron DOS Prediction after building electron DOS dataset.   

### Models
#### embedder eDOS
`DOSTransformer.py`: Our proposed model: DOSTransformer for Electron DOS

#### embedder phDOS
`DOSTransformer_phonon.py`: Our proposed model: DOSTransformer for Phonon DOS  


### Hyperparameters  

`--beta:` Hyperparameter for training loss controlling system_rmse (Balancing Term for Training)

`--layers:` Number of GNN layers in DOSTransformer model  

`--attn_drop:` Dropout ratio of attention weights

`--transformer:` Number of Transformer layer in DOSTransformer   

`--embedder:` Selecting embedder   

`--hidden:` Size of hidden dimension

`--epochs:`  Number of epochs for training the model

`--lr:` Learning rate for training the model  

`--dataset:` Selecting dataset for eDOS prediction (Random split, Crystal OOD, Element OOD, default dataset is Random split)

`--es:` Early Stopping Criteria  

`--eval:` Evaluation Step  
