# Zero Shot Reinforcement Learning from Low Quality Data


<a href="https://github.com/enjeeneer/zero-shot-rl/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![PyTorch](https://img.shields.io/badge/PyTorch-grey.svg?logo=PyTorch)](https://pytorch.org/get-started/pytorch-2.0/) [![Paper](http://img.shields.io/badge/paper-arxiv.2309.15178-B31B1B.svg)](https://arxiv.org/abs/2309.15178)

<img src="/media/vcfb-intuition-final.png" width=85% height=auto class="center">

_Figure 1: Conservative zero-shot RL methods suppress the values or measures on actions not in the  dataset for all tasks. Black dots represent state-action samples present in the dataset._

The is the official codebase for [Zero-Shot Reinforcement Learning from Low Quality Data](https://arxiv.org/abs/2309.15178) by [Scott Jeen](https://enjeeneer.io/), [Tom Bewley](https://tombewley.com/) and [Jonathan Cullen](http://www.eng.cam.ac.uk/profiles/jmc99).

## Summary

Imagine you've collected a dataset from a system you'd like to control more efficiently. Examples include: household robots, chemical manufacturing processes, autonomous vehicles,
or steel-making furnaces. An ideal solution would be to train an autonomous agent on your dataset, then for it to use what it learns to solve _any_ task inside the system. For our household robot, such
tasks may include sweeping the floor, making a cup of tea, or cleaning the windows. Formally, we call this problem setting _zero-shot reinforcement learning (RL)_, and taking steps toward realising it in the real-world is the focus of this work.

If our dataset is pseudo-optimal, that is to say, it tells our domestic robot the full extent of the floorspace, where the tea bags are stored, and how many windows exist,
then the existing state-of-the-art method, Forward Backward (FB) representations, performs excellently. On average it will
solve any task you want inside the system with 85% accuracy. However, if the data we've collected from the system is _suboptimal_--it doesn't provide all the information required to solve all tasks--then
FB representations fail. They fail because they overestimate the value of the data not present in the dataset, or in RL parlance, they
_overestimate out-of-distribution state-action values_--Figure 1 (Middle).

In this work, we resolve this by artificially suppressing these out-of-distribution values, leveraging ideas from _conservatism_ in the Offline RL literature--Figure 1 (Right). In experiments across
a variety of systems and tasks, we show these methods consistently outperform their non-conservative counterparts when the datasets are suboptimal--Figure 2.

<img src="/media/performance-profiles-subplot1.png" width=85% height=auto class="center">


_Figure 2: **Aggregate performance.** (Left) Normalised average performance w.r.t. single-task baseline algorithm CQL. (Right) Performance profiles showing distribution of scores across all tasks and domains. Both conservative FB variants stochastically dominate vanilla FB._


We also find that our proposals don't sacrifice performance when the dataset is pseudo-optimal, and so present little downside over their predecessors.


For further detail we recommend reading the paper. Direct any correspondance to [Scott Jeen](https://enjeeneer.io) or raise an issue!

## Setup
### Dependencies
Assuming you have [MuJoCo](https://mujoco.org/) installed, setup a conda env with [Python 3.9.16](https://www.python.org/downloads/release/python-3916/) using `requirements.txt` as usual:
```
conda create --name zsrl python=3.9.16
```
then install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

### Domains and Datasets
In the paper we report results with agents trained on datasets collected from different exploratory algorithms on different domains. The domains are:

| **Domain** | **Eval Tasks**                                                              | **Dimensionality** | **Type**      | **Reward** | **Command Line Argument** |
|--------------|-----------------------------------------------------------------------------|--------------------|---------------|-----------|---------------------------|
| Walker | `stand` `walk` `run` `flip`                                                 | Low                | Locomotion         | Dense     | `walker`                  |
| Quadruped | `stand` `roll` `roll_fast` `jump` `escape`                                  | High               | Locomotion         | Dense     | `quadruped`               |
| Point-mass Maze | `reach_top_left` `reach_top_right` `reach_bottom_left` `reach_bottom_right` | Low                | Goal-reaching      | Sparse    | `point_mass_maze`         |
| Jaco | `reach_top_left` `reach_top_right` `reach_bottom_left` `reach_bottom_right` | High               | Goal-reaching      | Sparse    | `jaco`                     |

and the dataset collecting algorithms are:

| **Dataset Collecting Algorithm**                                      | **State Coverage** | **Command Line Argument** |
|-----------------------------------------------------------------------|---------------------------|--------------------------|
 | [Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894) | High                      | `rnd`                    |
 | [Diversity is All You Need (DIAYN)](https://arxiv.org/abs/1802.06070)                                 | Medium                    | `diayn`                  |
 | Random                                                                | Low                       | `random`                 |

State coverage illustrations on `point_mass_maze` are provided in Figure 3. For each domain, datasets need to be downloaded manually from the [ExORL benchmark](https://github.com/denisyarats/exorl/tree/main) then reformatted. 
To download the `rnd` dataset on the `walker` domain, seperate their command line args with an `_` and run:  

```bash
python exorl_reformatter.py walker_rnd
```

this will create a single `dataset.npz` file in the `dataset/walker/rnd/buffer` directory.

<img src="/media/dataset-heatmap.png" width=70% height=auto class="center">

_Figure 3: **State coverage** by dataset on `point_mass_maze`._

### WandB
To use [Weights & Biases](https://wandb.ai/home) for logging, create a free account and run `wandb login` from the command line. 
Subsequent runs will automatically log to a new project named `conservative-world-models`.

### Algorithms
We provide implementations of the following algorithms: 

| **Algorithm**                                                               | **Authors**                                                  | **Command Line Argument** |
|-----------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------|
 | Conservative $Q$-learning                                                   | [Kumar et. al (2020)](https://arxiv.org/abs/2006.04779)      | `cql`                    |
 | Offline TD3                                                                 | [Fujimoto et. al (2021)](https://arxiv.org/pdf/2106.06860.pdf) | `td3`                    |
| Universal Successor Features learned with Laplacian Eigenfunctions (SF-LAP) | [Borsa et. al (2018)](https://arxiv.org/abs/1812.07626)      | `sf-lap`                 |
 | FB Representations                                                          | [Touati et. al (2023)](https://arxiv.org/abs/2209.14935)     | `fb`                     |
 | Value-Conservative FB Representations                                       | [Jeen et. al (2024)](https://arxiv.org/abs/2309.15178)       | `vcfb`                   |
 | Measure-Conservative FB Representations                                     | [Jeen et. al (2024)](https://arxiv.org/abs/2309.15178)             | `mcfb`                   |

### Training
To train a standard Value-Conservative Forward Backward Representation with the `rnd` (100k) dataset to solve all tasks in the `walker` domain, run:
```bash
python main_offline.py vcfb walker rnd --eval_task stand run walk flip
```

### Citation

If you find this work informative please consider citing the paper!


```commandline
@article{jeen2023,
  url = {https://arxiv.org/abs/2309.15178},
  author = {Jeen, Scott and Bewley, Tom and Cullen, Jonathan M.},  
  title = {Zero-Shot Reinforcement Learning from Low Quality Data},
  publisher = {arXiv},
  year = {2023},
}
```

## License 
This work licensed under a standard MIT License, see `LICENSE.md` for further details.





