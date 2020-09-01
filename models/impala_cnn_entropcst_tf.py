from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()

import numpy as np

def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x


class ImpalaCNNEntropCst(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self._timesteps_total = model_config['custom_options']['timesteps_total']
        self._current_sample_count = 0
        self._min_entrop_init_val = model_config['custom_options']['min_entrop_init']
        self._shan_entrop_init_val = model_config['custom_options']['shan_entrop_init']
        self._explore_time = model_config['custom_options']['explore_time']
        self._num_outputs = num_outputs
        self._max_min_entrop = 1. / num_outputs
        self._max_shan_entrop = np.log(num_outputs)

        depths = [16, 32, 32]

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)

        self._min_entrop_min_val = tf.keras.backend.variable(value=self._min_entrop_init_val * self._max_min_entrop, name='min_entrop_min_val')
        self._min_entrop_min_val._trainable = False
        self._shan_entrop_min_val = tf.keras.backend.variable(value=self._shan_entrop_init_val * self._max_shan_entrop, name='shan_entrop_min_val')
        self._shan_entrop_min_val._trainable = False

        logits_unproj = tf.keras.layers.Dense(units=num_outputs)(x)
        probs_unproj = tf.nn.softmax(logits_unproj)

        # Min entrop proj
        curr_entrop = tf.reduce_min(probs_unproj, axis=-1)
        violating_states = tf.less(curr_entrop, self._min_entrop_min_val - 1e-6)
        eta = tf.expand_dims((self._max_min_entrop - self._min_entrop_min_val) / tf.clip_by_value(self._max_min_entrop - curr_entrop, clip_value_min=1e-16, clip_value_max=1.), dim=-1)
        probs_min_proj = tf.where(violating_states, x=eta * probs_unproj + (1. - eta) * self._max_min_entrop, y=probs_unproj)

        clipped_probs_min = tf.clip_by_value(probs_min_proj, clip_value_min=1e-16, clip_value_max=1 - 1e-16)
        logits_min_proj = tf.math.log(clipped_probs_min)

        # Shannon entrop proj
        curr_entrop = tf.reduce_sum(-probs_min_proj * logits_min_proj, axis=-1)
        violating_states = tf.less(curr_entrop, self._shan_entrop_min_val - 1e-2)
        eta = tf.expand_dims((self._max_shan_entrop - self._shan_entrop_min_val) / tf.clip_by_value(self._max_shan_entrop - curr_entrop, clip_value_min=1e-16, clip_value_max=self._max_shan_entrop), dim=-1)
        proj_probs_shan = tf.where(violating_states, x=eta * probs_min_proj + (1. - eta) * self._max_min_entrop, y=probs_min_proj)

        clipped_probs = tf.clip_by_value(proj_probs_shan, clip_value_min=1e-16, clip_value_max=1 - 1e-16)
        logits = tf.math.log(clipped_probs, name="pi")

        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables + [self._min_entrop_min_val, self._shan_entrop_min_val])

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])

    def update_entropy_targets(self, sample_count, sess):
        self._current_sample_count += sample_count
        new_min_entrop_min_val = (1. - (self._current_sample_count / self._timesteps_total / self._explore_time)) * self._min_entrop_init_val * self._max_min_entrop
        new_shan_entrop_min_val = (1. - (self._current_sample_count / self._timesteps_total / self._explore_time)) * self._shan_entrop_init_val * self._max_shan_entrop
        self._min_entrop_min_val.load(new_min_entrop_min_val, session=sess)
        self._shan_entrop_min_val.load(new_shan_entrop_min_val, session=sess)
        print('min_entrop_cst {} shan_entrop_cst {}'.format(new_min_entrop_min_val, new_shan_entrop_min_val))

# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_entropcst_tf", ImpalaCNNEntropCst)
