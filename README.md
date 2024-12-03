# Zero Shot Reinforcement Learning from Low Quality Data

## NeurIPS 2024
<a href="https://github.com/enjeeneer/zero-shot-rl/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 [![Paper](http://img.shields.io/badge/paper-arxiv.2309.15178-B31B1B.svg)](https://arxiv.org/abs/2309.15178)

<img src="/media/vcfb-intuition-final.png" width=85% height=auto class="center">

_Figure 1: Conservative zero-shot RL methods suppress the values or measures on actions not in the  dataset for all tasks. Black dots represent state-action samples present in the dataset._

The is the official codebase for [Zero-Shot Reinforcement Learning from Low Quality Data](https://arxiv.org/abs/2309.15178) by [Scott Jeen](https://enjeeneer.io/), [Tom Bewley](https://tombewley.com/) and [Jonathan Cullen](http://www.eng.cam.ac.uk/profiles/jmc99).

## Summary

This work proposes methods for performing zero-shot RL when the pre-training datasets are small and homogeneous. 
We show that by suppressing the predicted values (or measures) for actions not in the dataset (Figure 1), we can resolve an overestimation bias that arises when the dataset is inexhaustive. We demonstrate this on the ExORL (Figure 2) and D4RL (Figure 3) benchmarks, showing improved performance over existing works.

<img src="/media/performance-profiles-subplot2.png" width=85% height=auto class="center">

_Figure 2: **Aggregate ExORL performance.** (Left) Normalised average performance w.r.t. single-task baseline algorithm CQL. (Right) Performance profiles showing distribution of scores across all tasks and domains. Both conservative FB variants stochastically dominate vanilla FB._

<img src="/media/d4rl-performance.png" width=50% height=auto class="center">

_Figure 3: **Aggregate D4RL performance.** Normalised average performance w.r.t. single-task baseline algorithm CQL._


For further detail check out the paper. Direct any correspondance to [Scott Jeen](https://enjeeneer.io) or raise an issue!

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

## Algorithms
We provide implementations of the following algorithms: 

| **Algorithm**                                                              | **Authors**                                                    | Type                   | **Command Line Argument** |
|----------------------------------------------------------------------------|----------------------------------------------------------------|------------------------|--------------------------|
 | Conservative $Q$-learning                                                  | [Kumar et. al (2020)](https://arxiv.org/abs/2006.04779)        | Single-task Offline RL | `cql`                    |
 | Offline TD3                                                                | [Fujimoto et. al (2021)](https://arxiv.org/pdf/2106.06860.pdf) | Single-task Offline RL | `td3`                    |
| Goal-conditioned Implicit $Q$-Learning (GC-IQL)                            | [Park et. al (2023)](https://arxiv.org/abs/2307.11949)         | Goal-conditioned RL    | `gciql`                  |
| Universal Successor Features learned with Laplacian Eigenfunctions (SF-LAP) | [Borsa et. al (2018)](https://arxiv.org/abs/1812.07626)        | Zero-shot RL           | `sf-lap`                 |
 | FB Representations                                                         | [Touati et. al (2023)](https://arxiv.org/abs/2209.14935)       |  Zero-shot RL                      | `fb`                     |
 | Value-Conservative FB Representations                                      | [Jeen et. al (2024)](https://arxiv.org/abs/2309.15178)         |  Zero-shot RL                      | `vcfb`                   |
 | Measure-Conservative FB Representations                                    | [Jeen et. al (2024)](https://arxiv.org/abs/2309.15178)         |  Zero-shot RL                      | `mcfb`                   |

## Domains and Datasets
### ExORL
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

State coverage illustrations on `point_mass_maze` are provided in Figure 4. For each domain, datasets need to be downloaded manually from the [ExORL benchmark](https://github.com/denisyarats/exorl/tree/main) then reformatted. 
To download the `rnd` dataset on the `walker` domain, seperate their command line args with an `_` and run:  

```bash
python exorl_reformatter.py walker_rnd
```

this will create a single `dataset.npz` file in the `dataset/walker/rnd/buffer` directory.

<img src="/media/dataset-heatmap.png" width=70% height=auto class="center">

_Figure 4: **State coverage** by dataset on `point_mass_maze`._

To train a standard Value-Conservative Forward Backward Representation with the `rnd` (100k) dataset to solve all tasks in the `walker` domain, run:
```bash
python main_exorl.py vcfb walker rnd --eval_task stand run walk flip
```

### D4RL
In the paper we report results on:

| **Domain**     | **Command Line Argument** |
|----------------|--------------------------|
| Walker         | `walker`                 |
| Cheetah        | `cheetah`                |

trained on the following datasets:

| **Datasets**  | **Description**                                                                                                                                                    | **Command Line Argument** |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
 | Medium        | Generated by training an SAC policy, early-stopping the training, and collecting 1M samples from this partially-trained policy                                     | `medium`                  |
 | Medium-replay | Generated by recording all samples in the replay buffer observed during training until the policy reaches the “medium” level of performance.                       | `medium-replay`           |
 | Medium-expert | Generated by mixing equal amounts of expert demonstrations and suboptimal data, either from a partially trained policy or by unrolling a uniform-at-random policy. | `medium-expert`           |

You'll need to manually download the D4RL datasets following their instructions, rename them to `dataset.hdf5` and place them in the correct directory inside `/datasets` e.g. the `walker` `medium-expert` dataset should be saved to `datasets/walker/medium-expert/dataset.hdf5`.
To train a standard Value-Conservative Forward Backward Representation with the `medium-expert` dataset on `walker`, run:
```bash
python main_d4rl.py vcfb walker medium-expert
```

## Citation

If you find this work informative please consider citing the paper!


```commandline
@article{jeen2024,
  url = {https://arxiv.org/abs/2309.15178},
  author = {Jeen, Scott and Bewley, Tom and Cullen, Jonathan M.},  
  title = {Zero-Shot Reinforcement Learning from Low Quality Data},
  journal = {Advances in Neural Information Processing Systems 38},
  year = {2024},
}
```

## License 
This work licensed under a standard MIT License, see `LICENSE.md` for further details.





