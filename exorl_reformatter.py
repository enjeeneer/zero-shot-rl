# pylint: disable=protected-access

"""Reformats the exorl dataset to a single .npz file."""

from argparse import ArgumentParser
from os import listdir
import numpy as np
from utils import BASE_DIR
from tqdm import tqdm
from loguru import logger
import subprocess

# Overwrite default config using argparse
parser = ArgumentParser()
parser.add_argument("domain_algorithm_pair", type=str)
args = parser.parse_args()

domain_algorithm_pair = [args.domain_algorithm_pair.rsplit("_", 1)]

print(domain_algorithm_pair)
# download from exorl bucket
for domain, algorithm in domain_algorithm_pair:

    subprocess.call(["bash", "download.sh", domain, algorithm])

    data_dir = BASE_DIR / f"datasets/{domain}/{algorithm}/buffer"
    video_dir = BASE_DIR / f"datasets/{domain}/{algorithm}/video"
    data_fnames = [f for f in listdir(data_dir) if f[-4:] == ".npz"]
    video_fnames = [f for f in listdir(video_dir) if f[-4:] == ".mp4"]
    new_dataset_path = BASE_DIR / f"datasets/{domain}/{algorithm}/dataset.npz"

    dataset = {}
    logger.info(f"Reformatting {domain} {algorithm} exorl dataset.")
    for fname in tqdm(data_fnames, desc="Reformatting dataset"):
        data = np.load(f"{data_dir}/{fname}")
        episode = {fname[:-4]: dict(data)}
        dataset = {**dataset, **episode}

    logger.info(f"Reformatting complete. Saving dataset to {new_dataset_path}.")
    np.savez(new_dataset_path, **dataset)

    # delete old files
    for fname in tqdm(data_fnames, desc="Deleting old buffer files"):
        (data_dir / fname).unlink()

    for fname in tqdm(video_fnames, desc="Deleting old video files"):
        (video_dir / fname).unlink()

    # delete old directories
    data_dir.rmdir()
    video_dir.rmdir()
