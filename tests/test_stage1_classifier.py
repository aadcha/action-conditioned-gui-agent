"""Unit tests for the Stage 1 MLP."""

import torch

from src.models.stage1_classifier import Stage1Classifier, Stage1Config


def test_forward_shape():
    model = Stage1Classifier(Stage1Config(feature_dim=64, hidden_dim=32, num_classes=8))
    x = torch.randn(7, 64)
    logits = model(x)
    assert logits.shape == (7, 8)


def test_num_trainable_params_nonzero():
    model = Stage1Classifier(Stage1Config(feature_dim=64, hidden_dim=32))
    assert model.num_trainable_params() > 0


def test_supports_overrides():
    # override via kwargs takes precedence over Stage1Config default
    model = Stage1Classifier(feature_dim=128, num_classes=4)
    x = torch.randn(3, 128)
    logits = model(x)
    assert logits.shape == (3, 4)
