#!/usr/bin/env python

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.memory import ray_get_and_free
from .policy import CustomPPOPolicy

class CustomSyncOptimizer(SyncSamplesOptimizer):
    @override(SyncSamplesOptimizer)
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            samples = []
            while sum(s.count for s in samples) < self.train_batch_size:
                if self.workers.remote_workers():
                    samples.extend(
                        ray_get_and_free([
                            e.sample.remote()
                            for e in self.workers.remote_workers()
                        ]))
                else:
                    samples.append(self.workers.local_worker().sample())
            samples = SampleBatch.concat_samples(samples)
            self.sample_timer.push_units_processed(samples.count)

        with self.grad_timer:
            for k, v in self.policies.items():
                v.model.update_entropy_targets(samples.count, v._sess)

            fetches = do_minibatch_sgd(samples, self.policies,
                                       self.workers.local_worker(),
                                       self.num_sgd_iter,
                                       self.sgd_minibatch_size,
                                       self.standardize_fields)
        self.grad_timer.push_units_processed(samples.count)

        if len(fetches) == 1 and DEFAULT_POLICY_ID in fetches:
            self.learner_stats = fetches[DEFAULT_POLICY_ID]
        else:
            self.learner_stats = fetches
        self.num_steps_sampled += samples.count
        self.num_steps_trained += samples.count
        return self.learner_stats


def custom_choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return CustomSyncOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"])


def get_policy_class(config):
    return CustomPPOPolicy


CustomPPO = PPOTrainer.with_updates(
    make_policy_optimizer=custom_choose_policy_optimizer,
    get_policy_class=get_policy_class,
)
