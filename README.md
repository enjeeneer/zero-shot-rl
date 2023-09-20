# Conservative World Models

---
Original implementations of Conservative Forward Backward representations as proposed in

[Conservative World Models](https://arxiv.org/abs/2110.00468) by 

[Scott Jeen](https://enjeeneer.io/), [Tom Bewley](https://tombewley.com/) & [Jonathan Cullen](http://www.eng.cam.ac.uk/profiles/jmc99)

## Method

<img src="/media/vcfb-intuition.png" width=100% height=auto>


## Setup
### Dependencies
Assuming you already have MuJoCo installed, install dependencies using `conda`:
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

![Dataset state coverage](/media/dataset-heatmap.png "Dataset state coverage on Maze")


For each domain-algorithm pair, their associated dataset needs to be downloaded manually from the [ExORL benchmark repo](https://github.com/denisyarats/exorl/tree/main) then reformatted. 
To download the `rnd` dataset on the `walker` domain, from the root run:  
```bash
./download.sh walker rnd
```
This will download the data as episodes of transitions into the `dataset/walker/rnd/buffer` directory.
Then, reformat the data into a single `.npz` file containing all transitions by running:
```bash
python exorl_reformatter.py walker rnd
```

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





