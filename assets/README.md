
## Requirements

We recommend using anaconda to manage the python environment. To create the environment for our benchmark, please follow the instruction as follows.

```bash
conda create -n $your_env_name
conda activate $your_env_name
```

install pytorch following the instruction on [pytorch installation](https://pytorch.org/get-started/locally/)

```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
```

intall pytorch-geometric following the instruction on [pyg installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```bash
conda install pyg -c pyg -c conda-forge
```

install the other dependencies

```bash
pip install ogb # package for ogb datasets
pip install texttable # show the running hyperparameters
pip install h5py # for Label Propagation
```

### Our Installation Notes for torch-geometric

What env configs that we tried that have succeeded: Mac/Linux + cuda driver 11.2 + Torch with cuda 11.1 + torch_geometric/torch sparse/etc with cuda 11.1.

What env configs that we tried by did not work: Linux + Cuda 11.1/11.0/10.2 + whatever version of Torch

In the above case when it did work, we adopted the following installation commands, and it automatically downloaded built wheels, and the installation completes within seconds. Installation codes that we adopted on Linux cuda 11.2 that did work:

```bash
  pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
  pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
  pip install torch-geometric
```

**Til now, you should be able to play with all of our implemented models except **Label Propagation**. To run LP, please follow our installation notes.**

### Installation guides for Julia (only required for certain modes of Label propagation, inherited from [C&S](https://github.com/CUAI/CorrectAndSmooth) )

First install Julia and PyJulia, following below instructions or instructions in https://pyjulia.readthedocs.io/en/latest/installation.html#install-julia

#### Installation guide for PyJulia on Linux:

Download Julia from official website, extract to whatever directory on your machine, there will be '/bin' at in the extracted folder.

```bash
export PATH=$PATH:/path-to-yout-extracted-julia/bin
```

After this step, type "julia", then you should be able to see Julia LOGO.

```bash
python3 -m pip install --user julia
```

use python to install

```python
>>> import julia
>>> julia.install()
```

activate julia and install requirements. To activate julia, until you see `julia> `, then type the following lines to install required packages in julia console:

```julia
import Pkg; Pkg.add("LinearMaps")
import Pkg; Pkg.add("Arpack")
import Pkg; Pkg.add("MAT")
```

## Train our soup ingredients and save the model state


```bash
python main.py --cuda_num=0  --type_model=$type_model --dataset=$dataset
# type_model in ['GraphSAGE', 'FastGCN', 'LADIES', 'ClusterGCN', 'GraphSAINT', 'SGC', 'SIGN', 'SIGN_MLP', 'LP_Adj', 'SAGN', 'GAMLP']
# dataset in ['Flickr', 'Reddit', 'Products', 'Yelp', 'AmazonProducts']
```

## Perform model soup across the soup ingredients using linear interpolation 

```
def merge_model(state_dicts):
    alphal = [1/len(state_dicts) for i in range(0, len(state_dicts))]
    sd = {}
    for k in state_dicts[0].keys():
        sd[k] = state_dicts[0][k].clone() * alphal[0]

    for i in range(1, len(state_dicts)):
        for k in state_dicts[i].keys():
            sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]

    return sd


def interpolate(state1, state2, model, data, split_idx, evaluator):
    alpha = np.linspace(0, 1, <granularity of interpolation>)
    max_val,  loc = -1,  -1
    for i in alpha:
        sd = {}
        for k in state1.keys():
            sd[k] = state1[k].clone() * i + state2[k].clone() * (1 - i)
        model.load_state_dict(sd)
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        if valid_acc > max_val:
            max_val = valid_acc
            loc = i
    sd = {}
    for k in state1.keys():
        sd[k] = state1[k].clone() * loc + state2[k].clone() * (1 - loc)
    return max_val, loc, sd
```
