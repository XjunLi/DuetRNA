# Remove only the bridge delta-distance features

This option removes the two distance-bin inputs derived from the bridge
displacements, while retaining the complete cross-edge branch. Existing
checkpoints load strictly without adding parameters or changing tensor shapes.

## Exact scope

Set `model.cross_edge.use_delta_distances=false` to zero only these input slots
before the cross-edge MLP:

- Current-frame bins of $\|\delta_i-\delta_j\|$, where $\delta_i=s_i-b_i$.
- The same distance bins computed from the self-conditioning predictions.

The cross-edge node features, residue-position encoding, optional relative
rotation input, learned MLP and biases all remain active and trainable. The
cross-edge output still enters `edge_fuse` and both IPA streams. Base-base and
sugar-sugar distance features, node embeddings, losses and decoding are unchanged.
There is no replacement geometry input and no new MLP.

The switch defaults to `true` when absent, preserving original checkpoint
behavior. Keeping the zeroed slots preserves the first linear layer's input
width and its saved weights. To inspect the implementation, see
[embeddings.py](../duetrna/models/embeddings.py) and
[flow_model.py](../duetrna/models/flow_model.py).

The earlier `model.cross_edge.enabled=false` whole-branch switch has been
removed. It now raises an explicit error instead of disabling the branch or
silently changing meaning. Remove that override, including from a saved run
configuration, and use `model.cross_edge.use_delta_distances=false`.

## Compare an existing checkpoint

Run from the repository root in the project environment, with the original
resolved training `config.yaml` beside the checkpoint. Keep seed, lengths and
sample counts identical, using separate output names.

```bash
python duetrna/inference.py \
  inference.ckpt_path=/absolute/path/to/checkpoint.ckpt \
  inference.name=bridge_distances_on inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.use_delta_distances=true
```

```bash
python duetrna/inference.py \
  inference.ckpt_path=/absolute/path/to/checkpoint.ckpt \
  inference.name=bridge_distances_off inference.seed=123 \
  inference.samples.length_subset='[80]' +inference.samples.samples_per_length=4 \
  inference.evalsuite.run_eval=false ++model.cross_edge.use_delta_distances=false
```

The double `++` adds the option to the inference config before it is merged over
the checkpoint's training config. The four-sample example is a smoke comparison.
Use the original IF/GS protocols and direct geometry metrics for conclusions
about quality. Removing these inputs changes a trained model's behavior; an
improvement is not assumed.

For training, use `model.cross_edge.use_delta_distances=false` and a new output
name with the existing training command. All parameter names, shapes, ordering
and trainability are preserved, including those in the cross-edge encoder.
Continuation from a checkpoint is a changed experiment and must retain its
other model and training settings.

## Regression checks

Run the tests without a dataset, checkpoint or GPU:

```bash
python -m pytest -q tests/test_bridge_ablation.py
```

They verify that only the two intended bridge input blocks change, base/sugar
edge inputs stay identical, the retained branch produces nonzero output and
receives gradients, and strict loading and optimizer-state restoration work.
They also cover legacy defaults, the retired switch, padding and global SE(3)
equivariance. Synthetic nonzero output weights prevent zero initialization from
hiding inactive paths. Actual checkpoint quality remains a separate evaluation.
