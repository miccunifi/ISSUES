# ISSUES (ICCVW 2023)

### Mapping Memes to Words for Multimodal Hateful Meme Classification

This is the **official repository** of the paper "*Mapp**I**ng Meme**S** to Word**S** for M**U**ltimodal Hateful M**E**me Cla**S**sification*" (**ISSUES**).

## Overview

>**Abstract**: <br>
> Hateful memes have recently taken a leading role in the proliferation of hate content on the Internet,
especially on social media, due to their expressiveness and subtlety that makes traditional systems
ineffective against them. Facebook AI has proposed a prize competition with a new carefully
crafted dataset (HMC) to stimulate research into artificial intelligence systems that can effectively
counter this harmful trend. Our work focuses on improving existing hateful memes classification
systems by leveraging the knowledge contained in large-scale pre-trained visual-language models
like CLIP. By using textual inversion to enrich the pre-trained model vocabulary with concepts associated with hate
speech, adapting the embedding spaces to the downstream task, and employing a combiner neural
network to learn an expressive fusion function that models the interaction between the two
modalities, we have outperformed all the existing baselines we considered. We performed extensive
experiments on the HMC and HarMeme datasets, obtaining an AUROC of 85.51 and 92.83 respectively on the HMC test-unseen split
and the HarMeme test set.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

We strongly recommend the use of the [**Anaconda**](https://www.anaconda.com/) package manager to avoid
dependency/reproducibility problems.
A conda installation guide for Linux systems can be
found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

### Installation

1. Clone the repository

```sh
git clone https://github.com/miccunifi/ISSUES.git
```

2. Install Python dependencies

Navigate to the root folder of the repository and use the command:
```sh
conda config --add channels conda-forge
conda create -n memes -y python=3.9.16
conda activate memes
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install --file requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Based on the CUDA version on your machine you may have to replace some package versions.

3. Log in to your WandB account
```sh
wandb login
```


## Datasets
We do not hold rights on the original HMC and HarMeme datasets. 
To download the full original datasets use the following links:


- HMC **[[link](https://hatefulmemeschallenge.com/)]** - Contains **12.140** memes
- HarMeme **[[link](https://github.com/di-dimitrov/mmf/tree/master/data/datasets/memes/defaults/images)]** - Contains **3.544** memes


### Data Preparation
Download the files in the [release](https://github.com/miccunifi/ISSUES/releases/tag/latest) and place the '_resources_' folder in the root of the project.

<pre>
project_base_path
└─── <b>resources</b>
  ...
└─── src
  | combiner.py
  | datasets.py
  | engine.py
  ...

...
</pre>

To work effectively with the code base, the HMC and HarMeme dataset images must be placed as the following structure:

<pre>
project_base_path
└─── resources
  └─── datasets
    └─── harmeme
      └─── clip_embds
          | test_no-proj_output.pt
          | train_no-proj_output.pt
          | val_no-proj_output.pt

      └─── <b>img
          | covid_memes_2.png
          | covid_memes_3.png
          | covid_memes_4.png
          ....</b>

      └─── labels
          | info.csv

    └─── hmc
      └─── clip_embds
          | dev_seen_no-proj_output.pt
          | dev_unseen_no-proj_output.pt
          | test_seen_no-proj_output.pt
          | test_unseen_no-proj_output.pt
          | train_no-proj_output.pt

      └─── <b>img
          | 01235.png
          | 01236.png
          | 01243.png
          ....</b>
        
      └─── labels
          | info.csv
  ...
  
└─── src
  | combiner.py
  | datasets.py
  | engine.py
  ...

...
</pre>

## Usage

Here's a brief description of each file under the ```src/``` directory:

* ```utils.py```: utils file
* ```combiner.py```: Combiner model definition
* ```textualInversion.py```: Textual Inversion model definition
* ```datasets.py```: Dataset code and collator definition
* ```engine.py```: Models definition for the multi-modal classification of hateful memes
* ```main.py```: main file


### Pre-trained models and weights

We provide the pre-trained checkpoint of our best models in the [release](https://github.com/miccunifi/ISSUES/releases/tag/latest).

To use the checkpoints to reproduce our results, they must be placed in the following structure:
<pre>
project_base_path
└─── resources
  └─── datasets
      ...
  └─── <b>pretrained_models
      | hmc_combiner_best.ckpt
      | hmc_text-inv-comb_best.ckpt
      | harmeme_combiner_best.ckpt
      | harmeme_text-inv-comb_best.ckpt</b>
      
  └─── pretrained_weights
      ...
  
└─── src
  | combiner.py
  | datasets.py
  | engine.py
  ...

...
</pre>

### Reproduce the results and run experiments
For running the following scripts in a decent amount of time, it is **heavily** recommended to use a CUDA-capable GPU.

In the root folder of the repository you can find the scripts to reproduce our results or train from scratch our models.
<pre>
project_base_path
└─── resources
  ...
  
└─── src
  ...

<b>run.sh
run_harmeme_combiner.sh
run_harmeme_text-inv-comb.sh
run_hmc_combiner.sh
run_hmc_text-inv-comb.sh
</b>

...
</pre>

To run these files, navigate to the root folder and use the following commands:

```shell
chmod +x <filename>.sh
./<filename>.sh
```

Disabling the ```--reproduce``` flag in one of the ```<filename>.sh``` files allows the training and evaluation of the model, whereas enabling it will use the
specified pre-trained checkpoint and evaluate the model on the test data only.

We recommend using the ```run.sh``` file to run experiments with different argument values.

## Arguments
We briefly describe some arguments of the scripts.
### Experiments
- ```dataset``` - the name of the dataset: [**hmc** or **harmeme**]
- ```num_mapping_layers``` - number of projection layers to map CLIP features in a task-oriented latent space
- ```num_pre_output_layers``` - number of MLP hidden layers for performing the final classification
- ```max_epochs``` - maximum number of epochs for the experiment
- ```lr``` - learning rate to use in the optimizer
- ```batch_size``` - size of the batch
- ```fast_process``` - flag to indicate whether to use pre-computed CLIP features as the input of the model instead of 
                        computing them during the training process
- ```name``` - name of the model to use
- ```pretrained_model``` - name of the checkpoint of a trained model in the 'pretrained_models' folder
- ```reproduce``` - flag to indicate whether to perform the training process followed by the evaluation phase (False) or directly evaluate a pre-trained model on the test data (True)

### General
- ```map_dim``` - output dimension of the projected feature vectors
- ```fusion``` - fusion method between the textual and visual modalities for some models: [**concat** or **align**]
- ```pretrained_proj_weights``` - flag to indicate whether to use pre-trained projection weights for some models
- ```freeze_proj_layers``` - flag to indicate whether to freeze the pre-trained weights


### Combiner Architecture
- ```comb_proj``` - flag to indicate whether to project the input features of the Combiner 
- ```comb_fusion``` - fusion method to use to combine the input features of the Combiner
- ```convex_tensor``` - flag to indicate whether to compute a tensor or a scalar as the output of the convex combination

### Textual Inversion Architecture
- ```text_inv_proj``` - flag to indicate whether to use CLIP textual encoder projection 
- ```phi_inv_proj``` - flag to indicate whether to project the output of phi network
- ```post_inv_proj``` - flag to indicate whether to project the CLIP textual encoder output features
- ```enh_text``` - flag to indicate whether to use a simple prompt with only the pseudo word or enhanced it 
by concatenating the memes' text
- ```phi_freeze``` - flag to indicate whether to freeze the pre-trained phi network 


---------------------------------------------------
## Authors

* [**Giovanni Burbi**](https://github.com/GiovanniBurbi)
* [**Alberto Baldrati**](https://scholar.google.it/citations?hl=en&user=I1jaZecAAAAJ)
* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number
101004545 - ReInHerit.

