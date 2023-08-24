# Graph Ladling [ICML 2023]
<img width="1034" alt="image" src="https://github.com/VITA-Group/graph_ladling/assets/6660499/6105c9b0-2e25-4791-b148-1ac3de0cc533">

## New Update

Our new ArXiv release include comparison with a concurrent work (Jiong et. al. 2023 https://arxiv.org/pdf/2305.09887.pdf) which independently presents similar ideas, among other SOTA-distributed GNN training works. 

More specifically, we summarize our key differences with Jiong et. al. as follows:

1. Our work optimizes model performance with/without distributed data parallelism by interpolating soup GNN candidate weights. On the other hand, Jiong et. al. 2023 improves performance for data-parallel GNN training with model averaging and randomized partition of graphs.
2. Our candiate models are interpolated only after training to facilitate diversity required for soup while Jiong et. al. weights are periodically averaged during training based on a time interval.
3. Our soup ingredients are trained by sampling different clusters per epoch on the full graph while Jiong et. al. individual trainers use localized subgraph assigned by randomized node/super-node partitions.

For more detailed discussion, please refer to Section 4.1 of our new ArXiv (https://arxiv.org/abs/2306.10466).



## Abstract

<img width="1160" alt="image" src="https://github.com/VITA-Group/graph_ladling/assets/6660499/abc2ed4c-37c5-4cce-9c5f-52cd8d07cf1d">


<img width="505" alt="image" src="https://github.com/VITA-Group/graph_ladling/assets/6660499/3fcccfdb-2186-4ef5-90fe-3cab41ffa991">

<img width="972" alt="image" src="https://github.com/VITA-Group/graph_ladling/assets/6660499/af2b41d9-d569-4fba-bfa5-9dfd836465f2">
