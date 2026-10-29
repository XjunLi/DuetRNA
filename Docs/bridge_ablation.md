# Disable bridge with an existing checkpoint

This guide explains how to compare an existing checkpoint with its delta
bridge enabled or disabled. The ablation adds no features or parameters.
It keeps checkpoint shapes intact and uses zero for the complete cross-edge
encoder output before edge fusion. Quality changes must be measured.

## What changes

`model.cross_edge.enabled` defaults to `true`, including for old training
configurations that lack this field. Setting it to `false` skips the bridge
encoder and replaces its output with zeros. Setting only the bridge's input
coordinates to zero would not disable it: its node features, residue-position
features and learned biases could still contribute.

| Component | Bridge enabled | Bridge disabled |
| --- | --- | --- |
| Cross-edge encoder output | Learned bridge features | Zero tensor |
| Edge fusion input | Base, sugar and bridge channels | Base, sugar and zero channels |
| Two IPA streams and object mixer | Existing implementation | Same implementation |
| Losses, decoder and sampler | Existing settings | Same settings |
| Checkpoint keys and parameter shapes | Original layout | Identical layout |

The implementation retains bridge modules so weights load with `strict=True`
in both modes. Skipped bridge parameters are frozen for training, while the
optimizer retains its existing parameter order. The fusion layer retains its
original dimensions and biases. This is an ablation of the bridge contribution,
not a smaller reparameterized network. It does not repair the missing relative
translation information by adding another input path.

See [flow_model.py](../duetrna/models/flow_model.py) for the switch and
[config.yaml](../duetrna/configs/config.yaml) for the default.

## Compare an existing checkpoint

Activate the existing project environment and run from the repository root.
Keep the original resolved training `config.yaml` beside the checkpoint.
Use separate output names and the same seed, lengths and sample count.

```bash
python duetrna/inference.py \
  inference.ckpt_path=/absolute/path/to/checkpoint.ckpt \
  inference.name=bridge_on inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.enabled=true
```

```bash
python duetrna/inference.py \
  inference.ckpt_path=/absolute/path/to/checkpoint.ckpt \
  inference.name=bridge_off inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.enabled=false
```

The double `++` is intentional: the inference YAML initially has no model
section. Its overrides are merged over the checkpoint's training configuration.
Both modes load the same weights strictly. The four-sample example is a smoke
comparison, not evidence of improved validity or diversity. Use the original
IF/GS protocols and direct geometry checks for the scientific comparison.

Disabling a branch in an already trained model changes its input distribution.
A drop in quality would show checkpoint dependence on the branch; it would not
prove that a model trained without it cannot work. Conversely, an unchanged
small sample does not prove the branch is unnecessary in general.

## Training and compatibility

For a new training run, use `model.cross_edge.enabled=false` with the existing
training command and a new `experiment.wandb.name`. No losses are removed.
For continuation from an existing checkpoint, retain its exact model options,
dataset and training configuration, set `experiment.warm_start` to that file,
and use a new output directory. State dictionaries and optimizer parameter
ordering remain compatible; actual distributed training should be checked in
the original environment. Continuation with bridge disabled is a new experiment,
not an unchanged resume of the original run.

## Run regression tests

These tests require PyTorch, NumPy, SciPy, OmegaConf and pytest, without data,
checkpoints, folding models or a GPU.

```bash
python -m pytest -q tests/test_bridge_ablation.py
```

They check legacy defaults, strict loading in both modes, equivalence to
zeroing the old encoder output, independence from disabled weights, global
SE(3) equivariance, padding, backward gradients and optimizer-state restoration.
Synthetic nonzero frame-update, IPA-output and edge-transition weights prevent
zero initialization from making the tests vacuous. These checks do not measure
model quality.
