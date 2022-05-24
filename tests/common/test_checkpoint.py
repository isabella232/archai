# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import yaml

from archai.common.checkpoint import CheckPoint
from archai.common.config import Config

CONFIG_FILEPATH = "config.yaml"
CHECKPOINT_FILENAME = "checkpoint.pt"
FREQ = 0


def _create_checkpoint_class():
    config_dict = {"filename": CHECKPOINT_FILENAME, "freq": FREQ}
    with open(CONFIG_FILEPATH, "w") as f:
        yaml.dump(config_dict, f)

    config = Config(config_filepath=CONFIG_FILEPATH)
    checkpoint = CheckPoint(config, False)

    return checkpoint


def test_checkpoint_class_init():
    checkpoint = _create_checkpoint_class()

    assert os.path.basename(checkpoint.filepath) == CHECKPOINT_FILENAME
    assert checkpoint.freq == FREQ
    assert isinstance(checkpoint._callbacks, list)

    os.remove(CONFIG_FILEPATH)


def test_checkpoint_class_load_existing():
    checkpoint = _create_checkpoint_class()

    checkpoint.load_existing()

    os.remove(CONFIG_FILEPATH)
