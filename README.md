# Introduction
This is a [PyTorch](http://pytorch.org/) implementation of the following research papers:
 * (1) [Hierarchical Text Generation and Planning for Strategic Dialogue](https://arxiv.org/abs/1712.05846)
 * (2) [Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://arxiv.org/abs/1706.05125)


The code is developed by [Facebook AI Research](http://research.fb.com/category/facebook-ai-research-fair).

The code trains neural networks to hold negotiations in natural language, and allows reinforcement learning self play and rollout-based planning.


# Citation
If you want to use this code in your research, please cite:
```
@inproceedings{DBLP:conf/icml/YaratsL18,
  author    = {Denis Yarats and
               Mike Lewis},
  title     = {Hierarchical Text Generation and Planning for Strategic Dialogue},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning,
               {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
               10-15, 2018},
  pages     = {5587--5595},
  year      = {2018},
  crossref  = {DBLP:conf/icml/2018},
  url       = {http://proceedings.mlr.press/v80/yarats18a.html},
  timestamp = {Fri, 13 Jul 2018 14:58:25 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/icml/YaratsL18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


# Dataset
We release our dataset together with the code, you can find it under `data/negotiate`. This dataset consists of 5808 dialogues, based on 2236 unique scenarios. Take a look at §2.3 of the paper to learn about data collection.

Each dialogue is converted into two training examples in the dataset, showing the complete conversation from the perspective of each agent. The perspectives differ on their input goals, output choice, and in special tokens marking whether a statement was read or written. See §3.1 for the details on data representation.
```
# Perspective of Agent 1
<input> 1 4 4 1 1 2 </input>
<dialogue> THEM: i would like 4 hats and you can have the rest . <eos> YOU: deal <eos> THEM: <selection> </dialogue>
<output> item0=1 item1=0 item2=1 item0=0 item1=4 item2=0 </output> 
<partner_input> 1 0 4 2 1 2 </partner_input>

# Perspective of Agent 2
<input> 1 0 4 2 1 2 </input>
<dialogue> YOU: i would like 4 hats and you can have the rest . <eos> THEM: deal <eos> YOU: <selection> </dialogue>
<output> item0=0 item1=4 item2=0 item0=1 item1=0 item2=1 </output>
<partner_input> 1 4 4 1 1 2 </partner_input>
```

# Setup
All code was developed with Python 3.0 on CentOS Linux 7, and tested on Ubuntu 16.04. In addition, we used PyTorch 1.0.0, CUDA 9.0, and Visdom 0.1.8.4.

We recommend to use [Anaconda](https://www.continuum.io/why-anaconda). In order to set up a working environment follow the steps below:
```
# Install anaconda
conda create -n py30 python=3 anaconda
# Activate environment
source activate py30
# Install PyTorch
conda install pytorch torchvision cuda90 -c pytorch
# Install Visdom if you want to use visualization
pip install visdom
```

# Usage
## Supervised Training

### Action Classifier
We use an action classifier to compare performance of various models. The action classifier is described in section 3 of (2). It can be trained by running the following command:
```
python train.py \
--cuda \
--bsz 16 \
--clip 2.0 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.1 \
--init_range 0.2 \
--lr 0.001 \
--max_epoch 7 \
--min_lr 1e-05 \
--model_type selection_model \
--momentum 0.1 \
--nembed_ctx 128 \
--nembed_word 128 \
--nhid_attn 128 \
--nhid_ctx 64 \
--nhid_lang 128 \
--nhid_sel 128 \
--nhid_strat 256 \
--unk_threshold 20 \
--skip_values \
--sep_sel \
--model_file selection_model.th
```

### Baseline RNN Model
This is the baseline RNN model that we describe in (1):
```
python train.py \
--cuda \
--bsz 16 \
--clip 0.5 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.1 \
--model_type rnn_model \
--init_range 0.2 \
--lr 0.001 \
--max_epoch 30 \
--min_lr 1e-07 \
--momentum 0.1 \
--nembed_ctx 64 \
--nembed_word 256 \
--nhid_attn 64 \
--nhid_ctx 64 \
--nhid_lang 128 \
--nhid_sel 128 \
--sel_weight 0.6 \
--unk_threshold 20 \
--sep_sel \
--model_file rnn_model.th
```

### Hierarchical Latent Model
In this section we provide guidelines on how to train the hierarchical latent model from (2). The final model requires two sub-models: the clustering model, which learns compact representations over intents; and the language model, which translates intent representations into language. Please read sections 5 and 6 of (2) for more details.

**Clustering Model**
```
python train.py \
--cuda \
--bsz 16 \
--clip 2.0 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.2 \
--init_range 0.3 \
--lr 0.001 \
--max_epoch 15 \
--min_lr 1e-05 \
--model_type latent_clustering_model \
--momentum 0.1 \
--nembed_ctx 64 \
--nembed_word 256 \
--nhid_ctx 64 \
--nhid_lang 256 \
--nhid_sel 128 \
--nhid_strat 256 \
--unk_threshold 20 \
--num_clusters 50 \
--sep_sel \
--skip_values \
--nhid_cluster 256 \
--selection_model_file selection_model.th \
--model_file clustering_model.th
```

**Language Model**
```
python train.py \
--cuda \
--bsz 16 \
--clip 2.0 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.1 \
--init_range 0.2 \
--lr 0.001 \
--max_epoch 15 \
--min_lr 1e-05 \
--model_type latent_clustering_language_model \
--momentum 0.1 \
--nembed_ctx 64 \
--nembed_word 256 \
--nhid_ctx 64 \
--nhid_lang 256 \
--nhid_sel 128 \
--nhid_strat 256 \
--unk_threshold 20 \
--num_clusters 50 \
--sep_sel \
--nhid_cluster 256 \
--skip_values \
--selection_model_file selection_model.th \
--cluster_model_file clustering_model.th \
--model_file clustering_language_model.th
```

**Full Model**
```
python train.py \
--cuda \
--bsz 16 \
--clip 2.0 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.2 \
--init_range 0.3 \
--lr 0.001 \
--max_epoch 10 \
--min_lr 1e-05 \
--model_type latent_clustering_prediction_model \
--momentum 0.2 \
--nembed_ctx 64 \
--nembed_word 256 \
--nhid_ctx 64 \
--nhid_lang 256 \
--nhid_sel 128 \
--nhid_strat 256 \
--unk_threshold 20 \
--num_clusters 50 \
--sep_sel \
--selection_model_file selection_model.th \
--lang_model_file clustering_language_model.th \
--model_file full_model.th
```

## Selfplay
If you want to have two pretrained models to negotiate against each another, use `selfplay.py`. For example, lets have two rnn models to play against each other:
```
python selfplay.py \
--cuda \
--alice_model_file rnn_model.th \
--bob_model_file rnn_model.th \
--context_file data/negotiate/selfplay.txt  \
--temperature 0.5 \
--selection_model_file selection_model.th
```
The script will output generated dialogues, as well as some statistics. For example:
```
================================================================================
Alice : book=(count:3 value:1) hat=(count:1 value:5) ball=(count:1 value:2)
Bob   : book=(count:3 value:1) hat=(count:1 value:1) ball=(count:1 value:6)
--------------------------------------------------------------------------------
Alice : i would like the hat and the ball . <eos>
Bob   : i need the ball and the hat <eos>
Alice : i can give you the ball and one book . <eos>
Bob   : i can't make a deal without the ball <eos>
Alice : okay then i will take the hat and the ball <eos>
Bob   : okay , that's fine . <eos>
Alice : <selection>
Alice : book=0 hat=1 ball=1 book=3 hat=0 ball=0
Bob   : book=3 hat=0 ball=0 book=0 hat=1 ball=1
--------------------------------------------------------------------------------
Agreement!
Alice : 7 points
Bob   : 3 points
--------------------------------------------------------------------------------
dialog_len=4.47 sent_len=6.93 agree=86.67% advantage=3.14 time=2.069s comb_rew=10.93 alice_rew=6.93 alice_sel=60.00% alice_unique=26 bob_rew=4.00 bob_sel=40.00% bob_unique=25 full_match=0.78 
--------------------------------------------------------------------------------
debug: 3 1 1 5 1 2 item0=0 item1=1 item2=1
debug: 3 1 1 1 1 6 item0=3 item1=0 item2=0
================================================================================
```

## Reinforcement Learning
To fine-tune a pretrained model with RL use the `reinforce.py` script:
```
python reinforce.py \
--cuda \
--alice_model_file rnn_model.th \
--bob_model_file rnn_model.th \
--output_model_file rnn_rl_model.th \
--context_file data/negotiate/selfplay.txt  \
--temperature 0.5 \
--verbose \
--log_file rnn_rl.log \
--sv_train_freq 4 \
--nepoch 4 \
--selection_model_file selection_model.th  \
--rl_lr 0.00001 \
--rl_clip 0.0001 \
--sep_sel
```

# License
This project is licenced under CC-by-NC, see the LICENSE file for details.
