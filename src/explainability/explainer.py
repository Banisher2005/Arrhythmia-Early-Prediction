"""
Unified Explainability Interface

Provides a single API for all explainability methods.

Supported Methods
-----------------
- GradCAM
- GradCAM++
- Saliency
- SmoothGrad
- Integrated Gradients

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

from typing import Optional

import torch

from src.explainability.gradcam import GradCAM1D
from src.explainability.gradcam_plus_plus import GradCAMPlusPlus
from src.explainability.integrated_gradients import IntegratedGradients
from src.explainability.saliency import SaliencyMap
from src.explainability.smoothgrad import SmoothGrad
from src.utils.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_METHODS = {
    "gradcam": GradCAM1D,
    "gradcam++": GradCAMPlusPlus,
    "gradcampp": GradCAMPlusPlus,
    "saliency": SaliencyMap,
    "smoothgrad": SmoothGrad,
    "integrated_gradients": IntegratedGradients,
    "ig": IntegratedGradients,
}


class Explainer:
    """
    Unified Explainability Interface.

    Example
    -------
    >>> explainer = Explainer(
    ...     model,
    ...     method="gradcam++"
    ... )
    ...
    >>> result = explainer.explain(ecg)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        method: str = "gradcam",
        device: Optional[torch.device] = None,
        **kwargs,
    ):

        method = method.lower()

        if method not in SUPPORTED_METHODS:

            raise ValueError(
                f"Unsupported explainability method: {method}\n"
                f"Available methods: {list(SUPPORTED_METHODS.keys())}"
            )

        self.method_name = method

        self.method = SUPPORTED_METHODS[
            method
        ](
            model=model,
            device=device,
            **kwargs,
        )

        logger.info(
            "Initialized Explainer using '%s'",
            method,
        )

    ###########################################################################

    def generate(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs,
    ):

        return self.method.generate(
            ecg=ecg,
            target_class=target_class,
            **kwargs,
        )

    ###########################################################################

    def explain(
        self,
        ecg: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs,
    ):

        return self.method.explain(
            ecg=ecg,
            target_class=target_class,
            **kwargs,
        )

    ###########################################################################

    def batch_generate(
        self,
        ecg_batch: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs,
    ):

        if not hasattr(
            self.method,
            "batch_generate",
        ):

            raise NotImplementedError(
                f"{self.method_name} "
                "does not support batch inference."
            )

        return self.method.batch_generate(
            batch=ecg_batch,
            target_class=target_class,
            **kwargs,
        )

    ###########################################################################

    def save(
        self,
        *args,
        **kwargs,
    ):

        if not hasattr(
            self.method,
            "save",
        ):

            raise NotImplementedError(
                f"{self.method_name} "
                "does not implement save()."
            )

        return self.method.save(
            *args,
            **kwargs,
        )

    ###########################################################################

    def overlay(
        self,
        *args,
        **kwargs,
    ):

        if not hasattr(
            self.method,
            "overlay",
        ):

            raise NotImplementedError(
                f"{self.method_name} "
                "does not implement overlay()."
            )

        return self.method.overlay(
            *args,
            **kwargs,
        )

    ###########################################################################

    @property
    def name(
        self,
    ) -> str:

        return self.method_name


###############################################################################
# Example
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

    logger.info("=" * 70)
    logger.info("Testing Unified Explainer")
    logger.info("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AMSRAN_GF().to(device)

    model.eval()

    ecg = torch.randn(
        1,
        1,
        180,
        device=device,
    )

    methods = [
        "gradcam",
        "gradcam++",
        "saliency",
        "smoothgrad",
        "integrated_gradients",
    ]

    for method in methods:

        logger.info(
            "Running %s...",
            method,
        )

        explainer = Explainer(
            model=model,
            method=method,
            device=device,
        )

        result = explainer.explain(
            ecg=ecg,
            show=False,
        )

        logger.info(
            "%s Prediction : %d",
            method,
            result["prediction"],
        )

        logger.info(
            "%s Confidence : %.4f",
            method,
            result["confidence"],
        )

    logger.info("=" * 70)
    logger.info("Unified Explainer Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()