# Bridge distance-feature ablation

`model.cross_edge.use_delta_distances` controls the bridge distance features
and defaults to `true`. Setting it to `false` zeros the distance-bin inputs
computed from $\|\delta_i-\delta_j\|$, where $\delta_i=s_i-b_i$, for both the
current frames and self-conditioning predictions. The cross-edge branch and
its other inputs remain active.

## Generation

Run from the repository root. Use the same checkpoint, seed, lengths, and
sample counts, with separate output names:

```bash
python -m duetrna.inference \
  inference.ckpt_path=/path/to/model.ckpt \
  inference.training_config_path=/path/to/config.yaml \
  inference.name=bridge_distances_on inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.use_delta_distances=true
```

```bash
python -m duetrna.inference \
  inference.ckpt_path=/path/to/model.ckpt \
  inference.training_config_path=/path/to/config.yaml \
  inference.name=bridge_distances_off inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.use_delta_distances=false
```

## Training

Add `model.cross_edge.use_delta_distances=false` to the
[training command](../README.md#training).
