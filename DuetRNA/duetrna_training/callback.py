import torch
from pytorch_lightning import Callback

class NanGradientCallback(Callback):
    def __init__(self, *, fail_on_nonfinite=False):
        super().__init__()
        self.fail_on_nonfinite = bool(fail_on_nonfinite)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Called right before an optimizer step to check gradients for non-finite values."""
        grad_info = self._first_nonfinite_gradient(pl_module)
        if grad_info is not None:
            param_name, nonfinite_count, finite_max_abs = grad_info
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "nan_gradient_detected": 1,
                        "nonfinite_gradient_count": float(nonfinite_count),
                        "nonfinite_gradient_finite_max_abs": float(finite_max_abs),
                    },
                    step=trainer.global_step,
                )
            message = (
                "Non-finite gradients detected at global step "
                f"{trainer.global_step}, first_param={param_name}, "
                f"nonfinite_count={nonfinite_count}, finite_max_abs={finite_max_abs:.4e}"
            )
            if self.fail_on_nonfinite:
                # Counting a cleared-gradient optimizer step as successful
                # silently shortens the effective training schedule. Formal
                # runs fail closed so the cause can be inspected and resumed.
                raise FloatingPointError(message)
            pl_module.zero_grad(set_to_none=True)
            print(f"{message}; clearing gradients")

    def _first_nonfinite_gradient(self, pl_module):
        """Return diagnostics for the first parameter with non-finite gradients."""
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            finite_grad = grad[finite_mask]
            finite_max_abs = float(finite_grad.abs().max().item()) if finite_grad.numel() > 0 else 0.0
            return name, int((~finite_mask).sum().item()), finite_max_abs
        return None
