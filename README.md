## LR-GNN: a graph neural network based on link representation for predicting molecular associations

### Overview
This repository contains codes necessary to run the LR-GNN algorithm. 

### Running Environment
* Windows environment, Python 3
* PyTorch >= 1.3.1

### Datasets
All datasets are available at [data](http://bioinfo.nankai.edu.cn/kangcz.html).

### Model Framework
![Model framework of LR-GNN](Workflow.png)
Fig.1. The architecture of LR-GNN. 
The layer number of illustrated LR-GNN is 3. f() is a activation function and Wx+b denotes the linear transformation. 
The graph structure and initial features are input into 3 layers GCN-encoder to obtain the node embedding. 
Based on the link samples, target node embedding of each layer is input into propagation rule to construct link representation of 3 layers.
Finally, the link representations of all layers are fused in output by layer-wise fusing rule to predict molecular associations.

### Contacts
Please send any questions you might have about the code and/or the algorithm to [kangchuanze@mail.nankai.edu.cn](kangchuanze@mail.nankai.edu.cn).

