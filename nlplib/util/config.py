"""
Example:

    TRAIN_CONFIG_DEFAULT = dict(
        optim=dict(
            gradient_checkpointing=True,
            optim="adamw_torch_fused"
        ),
        training=dict(
            num_train_epochs=2,
            train_batch_size=16,
            learning_rate=1.0e-5,
            warmup_steps=0.1
        ),
        seed=42
    )

    args["train_config"] = ...
    train_config = merge_configs(TRAIN_CONFIG_DEFAULT, args["train_config"])
"""

import copy


_TRAIN_CONFIG_ALIAS = dict(
    per_device_train_batch_size=['train_batch_size', 'batch_size']
)

_ALIAS_TO_CANONICAL = {
    alias: name
    for name, aliases in _TRAIN_CONFIG_ALIAS.items()
    for alias in aliases
}


def normalize(config):
    result = {}
    for key, value in config.items():
        canonical = _ALIAS_TO_CANONICAL.get(key, key)
        if canonical in result:
            raise KeyError(f'Conflicting aliases for {canonical!r}')
        result[canonical] = normalize(value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_configs(*configs):
    # конфиги накладываются слева направо: последний имеет наивысший приоритет
    result = {}
    for config in configs:
        if config:
            deep_update(result, normalize(config))
    return result
