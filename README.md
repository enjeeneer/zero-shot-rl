# Conservative World Models

Original implementations of Conservative Forward Backward representations as proposed in

[Conservative World Models](https://arxiv.org/abs/2110.00468)  

[//]: # ([Scott Jeen]&#40;https://enjeeneer.io/&#41;, [Tom Bewley]&#40;https://tombewley.com/&#41; & [Jonathan Cullen]&#40;http://www.eng.cam.ac.uk/profiles/jmc99&#41;)

## Summary
Imagine you've collected a dataset from a system you'd like to control more efficiently. Examples of such systems include: household robots, chemical manufacturing processes, autonomous vehicles,
or steel-making furnaces. An ideal solution would be to train an autonomous agent on your dataset, then for it to use what it learns to solve _any_ task inside the system. For our household robot, such
tasks may include sweeping the floor, making a cup of tea, or cleaning the windows. Formally, we call this problem setting _zero-shot reinforcement learning (RL)_, and taking steps toward realising it in the real-world is the focus of this work.

If our dataset is pseudo-optimal, that is to say, it tells our domestic robot the full extent of the floorspace, where the tea bags are stored, and how many windows exist,
then the existing state-of-the-art method, Forward Backward (FB) representations, performs excellently. On average it will 
solve any task you want inside the system with 85% accuracy. However, if the data we've collected from the system is _suboptimal_--it doesn't provide all the information required to solve all tasks--
then FB representations fail. They fail because they overestimate the value of the data not present in our dataset, or in RL parlance, they 
_overestimate out-of-distribution state-action values_. We illustrate this below in the middle plot:

<figure>
 <img src="/media/vcfb-intuition.png" width=70% height=auto class="center">
 <figcaption>Figure 1. Test </figcaption>
</figure>

In this work, we resolve this by artifically suppressing these out-of-distribution values, leveraging so-called _conservatism_ from the Offline RL literature.
The family of algorithms we propose are called _Conservative_ Forward Backward representations, illustrated in the right plot above. In experiments across
a variety of systems, tasks, we show these methods consistently outperform FB representations when the datasets are suboptimal, see below:

<img src="/media/performance-profiles-subplot.png" width=70% height=auto class="center">

We also find that our proposals don't sacrifice performance when the dataset is pseudo-optimal, so present little downside over their predecessor.

For further detail we recommend reading the paper. Direct any correspondance to [Scott Jeen](https://enjeeneer.io) or raise any issue on this repo!

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

| **Algorithm**                           | **Authors**                                                    | **Command Line Argument** |
|-----------------------------------------|----------------------------------------------------------------| -------------------------------|
 | Conservative $Q$-learning               | [Kumar et. al (2020)](https://arxiv.org/abs/2006.04779)        | `cql`|
 | Offline TD3                             | [Fujimoto et. al (2021)](https://arxiv.org/pdf/2106.06860.pdf) | `td3`|
 | FB Representations                      | [Touati et. al (2022)](https://arxiv.org/abs/2209.14935)                                       | `fb`|
 | Value-Conservative FB Representations   | Jeen et. al (2023)                                             | `vcfb`|
 | Measure-Conservative FB Representations | Jeen et. al (2023)                                             | `mcfb`|

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





