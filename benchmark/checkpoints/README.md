Place the shared RhoFold benchmark checkpoint here. Checkpoints are ignored by
Git; only this instruction file belongs in the repository.

Expected files:

- `RhoFold_pretrained.pt`

Download the same checkpoint used by the RNA-FrameFlow evaluation protocol:

```bash
mkdir -p benchmark/checkpoints
wget \
  "https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/api/download?filename=RhoFold_pretrained.pt" \
  -O benchmark/checkpoints/RhoFold_pretrained.pt
```

Keep this endpoint and filename for RNA-FrameFlow-compatible evaluation.
