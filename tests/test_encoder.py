import itertools
import math
import random
import os

from pytest import approx
import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(1, os.getcwd())
from src.encoder import Encoder

class TestEncoder():
    def test_minimal(self):
        num_words, num_tags, num_chars = 10, 10, 100
        encoder = Encoder(num_words, num_tags, num_chars=num_chars)

        assert encoder.num_tags == num_tags
        assert isinstance(encoder.word_embedding, nn.Embedding)