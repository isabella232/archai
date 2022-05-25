# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.common.cocob import CocobBackprop, CocobOns

PARAMS = [torch.tensor([1.0], requires_grad=True)]
ALPHA = 100.0
EPS = 1e-8


def test_cocob_backprop_class_init():
    cocob_backprop = CocobBackprop(PARAMS, ALPHA, EPS)

    assert cocob_backprop.alpha == ALPHA
    assert cocob_backprop.eps == EPS


def test_cocob_backprop_class_step():
    cocob_backprop = CocobBackprop(PARAMS, ALPHA, EPS)

    loss = cocob_backprop.step()

    assert loss is None


def test_cocob_ons_class_init():
    cocob_ons = CocobOns(PARAMS, EPS)

    assert cocob_ons.eps == EPS


def test_cocob_ons_class_step():
    cocob_ons = CocobOns(PARAMS, EPS)

    loss = cocob_ons.step()

    assert loss is None
