# Environment

```bash
conda create -n glg

conda activate glg

conda install python=3.10

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install torch-explain

pip install ipykernel ipywidgets matplotlib snakeviz gensim smart_open plotly tensorboard cython numba scikit-image networkx pgmpy polyleven scikit-learn-extra torchaudio wandb kaleido
```
