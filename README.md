# Conservative World Models

---
Original implementations of Conservative Forward Backward representations as proposed in

[Conservative World Models](https://arxiv.org/abs/2110.00468) by 

[Scott Jeen](https://enjeeneer.io/), [Tom Bewley](https://tombewley.com/) & [Jonathan Cullen](http://www.eng.cam.ac.uk/profiles/jmc99)

## Method

This work focuses on performing zero-shot reinforcement learning (RL) from suboptimal datasets. In zero-shot RL, we assume the agent
has access to a dataset of transitions collected from the environment that it can use to build a world model to train its policy against (below (_left_)).
The existing state-of-the-art method, Forward Backward (FB) representations, does this remarkably well when provided access to large and diverse datasets. 
However, when the dataset is small or collected from a suboptimal behaviour policy, FB representations fail, specifically by overestimating the value of actions not in the dataset (below (_middle_)).

<img src="/media/vcfb-intuition.png" width=70% height=auto class="center">

As a fix, we propose a family of _Conservative_ Forward Backward representations, which suppresses the value of actions not in the dataset (above (_right_))

<img src="/media/performance-profiles-subplot.png" width=70% height=auto class="center">

In experiments across a variety of domains, tasks and datasets, we show our family of conservative algorithms performs favourably w.r.t vanilla FB (above). For more details
we direct the reader to the paper linked above.

## Setup
### Dependencies
Assuming you have [MuJoCo](https://mujoco.org/) installed, install dependencies using `conda`:
```
conda env create -f environment.yaml
conda activate world-models
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

As an illustrative example, the state coverage on `point_mass_maze` looks like this:

<img src="/media/dataset-heatmap.png" width=70% height=auto class="center">

For each domain, dataset need to be downloaded manually from the [ExORL benchmark](https://github.com/denisyarats/exorl/tree/main) then reformatted. 
To download the `rnd` dataset on the `walker` domain, seperate their command line args with an `_` and run:  

```bash
python exorl_reformatter.py walker_rnd
```

this will create a single `dataset.npz` file in the `dataset/walker/rnd/buffer` directory.

### WandB
To use [Weights & Biases](https://wandb.ai/home) for logging, create a free account and run `wandb login` from the command line. 
Then, set the `WANDB_PROJECT` variable in `utils.py` to the name of the project you want to log to.
Subsequent runs should automatically log to this project.

### Algorithms
We provide implementations of the following algorithms: 

| **Algorithm**                                         | **Authors**                                                    | **Command Line Argument** |
|-------------------------------------------------------|----------------------------------------------------------------| -------------------------------|
 | Conservative $Q$-learning                             | [Kumar et. al (2020)](https://arxiv.org/abs/2006.04779)        | `cql`|
 | Offline TD3                                           | [Fujimoto et. al (2021)](https://arxiv.org/pdf/2106.06860.pdf) | `td3`|
 | Forward-Backward Representations                      | [Touati et. al (2022)](https://arxiv.org/abs/2209.14935)                                       | `fb`|
 | Value-Conservative Forward-Backward Representations   | Jeen et. al (2023)                                             | `vcfb`|
 | Measure-Conservative Forward-Backward Representations | Jeen et. al (2023)                                             | `mcfb`|

### Training
To train a standard Value-Conservative Forward Backward Representation with the `rnd` dataset to solve all tasks in the `walker` domain, run:
```bash
python main_offline.py vcfb walker rnd --eval_task stand run walk flip
```

### Finetuning


## Citation
If you found this work useful, or you use this project to inform your own research, please consider citing it with:
```commandline
@article{jeen2022,
  url = {https://arxiv.org/abs/2206.14191},
  author = {Jeen, Scott R. and Abate, Alessandro and Cullen, Jonathan M.},  
  title = {Low Emission Building Control with Zero Shot Reinforcement Learning},
  publisher = {arXiv},
  year = {2022},
}
```

## License 
This work licensed under a standard MIT License, see `LICENSE.md` for further details.





