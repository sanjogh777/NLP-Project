# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl  torchvision
#import torch

!python3 -m nltk.downloader punkt

!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

!mkdir squad
!mkdir squad/dataset
!cp train-v1.1.json squad/dataset
!cp dev-v1.1.json squad/dataset
!ls squad/dataset

!wget http://nlp.stanford.edu/data/glove.840B.300d.zip
!unzip glove.840B.300d.zip

!pip install --upgrade gensim

!python -m gensim.scripts.glove2word2vec --input  glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt