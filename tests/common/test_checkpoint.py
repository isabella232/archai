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


def test_checkpoint_class_load_existing_false():
    checkpoint = _create_checkpoint_class()

    assert checkpoint.load_existing() is False

    os.remove(CONFIG_FILEPATH)


def test_checkpoint_class_load_existing_true():
    checkpoint = _create_checkpoint_class()

    checkpoint["dummy"] = "dummy"
    checkpoint.commit()
    checkpoint.clear()

    assert checkpoint.load_existing() is True

    os.remove(CONFIG_FILEPATH)
    os.remove(CHECKPOINT_FILENAME)


def test_checkpoint_class_new():
    checkpoint = _create_checkpoint_class()

    checkpoint.new()

    os.remove(CONFIG_FILEPATH)


def test_checkpoint_class_commit():
    checkpoint = _create_checkpoint_class()
    checkpoint["dummy"] = "dummy"

    checkpoint.commit()

    os.remove(CONFIG_FILEPATH)
    os.remove(CHECKPOINT_FILENAME)


def test_checkpoint_class_is_empty():
    checkpoint = _create_checkpoint_class()

    assert checkpoint.is_empty() is True

    os.remove(CONFIG_FILEPATH)
