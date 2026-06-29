"""
Explainability Benchmark

Benchmarks explainability algorithms.

Compares

- GradCAM
- GradCAM++
- Saliency
- SmoothGrad
- Integrated Gradients

Measures

- Runtime
- Peak GPU Memory
- Throughput
- Average Latency
- FPS

Author
------
Arrhythmia Early Prediction Project
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.explainability.explainer import Explainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityBenchmark:

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        warmup_runs: int = 5,
        benchmark_runs: int = 50,
    ):

        self.model = model.eval()

        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )

        self.warmup_runs = warmup_runs

        self.benchmark_runs = benchmark_runs

        self.methods = [
            "gradcam",
            "gradcam++",
            "saliency",
            "smoothgrad",
            "integrated_gradients",
        ]

    ###########################################################################

    def _benchmark_method(
        self,
        ecg: torch.Tensor,
        method: str,
    ):

        explainer = Explainer(
            model=self.model,
            method=method,
            device=self.device,
        )

        ##########################################################

        for _ in range(
            self.warmup_runs
        ):

            explainer.generate(
                ecg,
            )

        ##########################################################

        runtimes = []

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        for _ in range(
            self.benchmark_runs
        ):

            start = time.perf_counter()

            explainer.generate(
                ecg,
            )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            end = time.perf_counter()

            runtimes.append(
                end - start
            )

        runtime = np.asarray(
            runtimes
        )

        average = runtime.mean()

        std = runtime.std()

        fps = 1.0 / average

        gpu_memory = None

        if torch.cuda.is_available():

            gpu_memory = (

                torch.cuda.max_memory_allocated()

                / (1024 ** 2)

            )

        return {

            "method": method,

            "mean_runtime_ms":
                average * 1000,

            "std_runtime_ms":
                std * 1000,

            "fps":
                fps,

            "gpu_memory_mb":
                gpu_memory,

        }

    ##########################################################################

    def benchmark(
        self,
        ecg: torch.Tensor,
    ):

        results = []

        logger.info(
            "Running explainability benchmark..."
        )

        for method in self.methods:

            logger.info(
                "Benchmarking %s",
                method,
            )

            results.append(

                self._benchmark_method(
                    ecg,
                    method,
                )

            )

        return results
    ##########################################################################

    def save_json(
        self,
        results,
        output_directory: str,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_directory
            / "benchmark.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
            )

        logger.info(
            "Saved JSON benchmark -> %s",
            output_file,
        )

    ##########################################################################

    def save_csv(
        self,
        results,
        output_directory: str,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_directory
            / "benchmark.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "Method",
                    "Mean Runtime (ms)",
                    "Std Runtime (ms)",
                    "FPS",
                    "GPU Memory (MB)",
                ]
            )

            for row in results:

                writer.writerow(
                    [
                        row["method"],
                        row["mean_runtime_ms"],
                        row["std_runtime_ms"],
                        row["fps"],
                        row["gpu_memory_mb"],
                    ]
                )

        logger.info(
            "Saved CSV benchmark -> %s",
            output_file,
        )

    ##########################################################################

    def plot_runtime(
        self,
        results,
        output_directory: str,
    ):

        methods = [
            r["method"]
            for r in results
        ]

        runtime = [
            r["mean_runtime_ms"]
            for r in results
        ]

        plt.figure(
            figsize=(8,5)
        )

        plt.bar(
            methods,
            runtime,
        )

        plt.ylabel(
            "Runtime (ms)"
        )

        plt.title(
            "Explainability Runtime Comparison"
        )

        plt.grid(
            axis="y"
        )

        plt.tight_layout()

        output_file = (
            Path(output_directory)
            / "runtime_comparison.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "Saved Runtime Plot -> %s",
            output_file,
        )

    ##########################################################################

    def plot_memory(
        self,
        results,
        output_directory: str,
    ):

        memory = [
            (
                r["gpu_memory_mb"]
                if r["gpu_memory_mb"] is not None
                else 0
            )
            for r in results
        ]

        methods = [
            r["method"]
            for r in results
        ]

        plt.figure(
            figsize=(8,5)
        )

        plt.bar(
            methods,
            memory,
        )

        plt.ylabel(
            "GPU Memory (MB)"
        )

        plt.title(
            "GPU Memory Usage"
        )

        plt.grid(
            axis="y"
        )

        plt.tight_layout()

        output_file = (
            Path(output_directory)
            / "memory_comparison.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "Saved Memory Plot -> %s",
            output_file,
        )
    ##########################################################################

    def save_summary(
        self,
        results,
        output_directory: str,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_directory
            / "summary.txt"
        )

        fastest = min(
            results,
            key=lambda x: x["mean_runtime_ms"],
        )

        slowest = max(
            results,
            key=lambda x: x["mean_runtime_ms"],
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            file.write("=" * 60 + "\n")
            file.write("Explainability Benchmark Summary\n")
            file.write("=" * 60 + "\n\n")

            for result in results:

                file.write(
                    f"Method              : {result['method']}\n"
                )

                file.write(
                    f"Mean Runtime (ms)   : {result['mean_runtime_ms']:.3f}\n"
                )

                file.write(
                    f"Std Runtime (ms)    : {result['std_runtime_ms']:.3f}\n"
                )

                file.write(
                    f"FPS                 : {result['fps']:.2f}\n"
                )

                file.write(
                    f"GPU Memory (MB)     : {result['gpu_memory_mb']}\n"
                )

                file.write("\n")

            file.write("-" * 60 + "\n")

            file.write(
                f"Fastest Method : {fastest['method']}\n"
            )

            file.write(
                f"Slowest Method : {slowest['method']}\n"
            )

        logger.info(
            "Saved Summary -> %s",
            output_file,
        )

    ##########################################################################

    def run(
        self,
        ecg: torch.Tensor,
        output_directory: str = (
            "results/explainability/benchmark"
        ),
    ):

        results = self.benchmark(
            ecg,
        )

        self.save_json(
            results,
            output_directory,
        )

        self.save_csv(
            results,
            output_directory,
        )

        self.plot_runtime(
            results,
            output_directory,
        )

        self.plot_memory(
            results,
            output_directory,
        )

        self.save_summary(
            results,
            output_directory,
        )

        logger.info(
            "Explainability benchmark completed."
        )

        return results


###############################################################################
# Standalone Test
###############################################################################

def main():

    from src.models.amsran_gf import AMSRAN_GF

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

    benchmark = ExplainabilityBenchmark(
        model=model,
        device=device,
        warmup_runs=5,
        benchmark_runs=50,
    )

    benchmark.run(
        ecg=ecg,
    )


if __name__ == "__main__":
    main()