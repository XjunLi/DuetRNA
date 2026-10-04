__all__ = ["EvalSuite", "DuetRNAEvalSuite"]


def __getattr__(name):
    if name == "EvalSuite":
        from .evalsuite import EvalSuite

        return EvalSuite
    if name == "DuetRNAEvalSuite":
        from .evalsuite import DuetRNAEvalSuite

        return DuetRNAEvalSuite
    raise AttributeError(name)
