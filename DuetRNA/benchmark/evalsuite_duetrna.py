from __future__ import annotations

from .evalsuite import EvalSuite


class DuetRNAEvalSuite(EvalSuite):
    """Joint/direct benchmark wrapper for dual-frame DuetRNA models."""

    def __init__(
        self,
        save_dir,
        paths=None,
        constants=None,
        gpu_id1=0,
        gpu_id2=1,
        use_invfold: bool = False,
        prefer_generated_fasta: bool = True,
        load_models: bool = True,
    ):
        super().__init__(
            save_dir=save_dir,
            paths=paths,
            constants=constants,
            gpu_id1=gpu_id1,
            gpu_id2=gpu_id2,
            use_invfold=use_invfold,
            prefer_generated_fasta=prefer_generated_fasta,
            load_models=load_models,
        )
