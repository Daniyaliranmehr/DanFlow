# training/checker.py

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .trainer import Trainer