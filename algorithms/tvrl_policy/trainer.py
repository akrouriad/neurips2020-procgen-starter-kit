#!/usr/bin/env python

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.trainer import COMMON_CONFIG
from .policy import TVRLPolicy

DEFAULT_CONFIG = COMMON_CONFIG.update(
    {
        'entropy_cst_type': 'min',
        'device': 'cpu',
        'exploration_time': .75,
        'entrop_min_init_ratio': .5,
        'timesteps_total': 8e6,
        'lambda_final': .5,
        'tv_max': .1,
        'lambda': .95,
        'sgd_minibatch_size': 64,
     }
)  # Default config parameters that can be overriden by experiments YAML.

TVRLPolicyTrainer = build_trainer(
    name="TVRLPolicyTrainer",
    default_policy=TVRLPolicy,
    default_config=DEFAULT_CONFIG,
)
