import logging
import numpy as np
import torch
from typing import Union, Callable, Any

from .worker import TorchWorker
from .server import TorchServer


class DistributedSimulatorBase(object):
    """Simulate distributed programs with low memory usage.

    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers

    This base class is used by both trainer and evaluator.
    """

    def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug
        self.workers = []

        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")


class ParallelTrainer(DistributedSimulatorBase):
    """Synchronous and parallel training with specified aggregator."""

    def __init__(
        self,
        server: TorchServer,
        aggregator_inner: Callable[[list], torch.Tensor],
        aggregator_outer: Callable[[list], torch.Tensor],
        pre_batch_hooks: list,
        post_batch_hooks: list,
        max_batches_per_epoch: int,
        log_interval: int,
        metrics: dict,
        use_cuda: bool,
        debug: bool,
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.aggregator_inner = aggregator_inner
        self.aggregator_outer = aggregator_outer
        self.server = server
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.log_interval = log_interval
        self.max_batches_per_epoch = max_batches_per_epoch
        self.omniscient_callbacks = []
        self.random_states = {}
        self.alpha = 0.9
        self.gradn = 0
        super().__init__(metrics, use_cuda, debug)

    def aggregation_and_update(self):
        # If there are Byzantine workers, ask them to craft attacks based on the updated models.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()

        gradients = self.parallel_get(lambda w: w.get_gradient())
        gradients_current = self.parallel_get(lambda w: w.get_current_gradient())

        aggregated = self.aggregator_inner(gradients)
        aggregated_current = self.aggregator_outer(gradients_current)
        #
        self.gradn = self.alpha * aggregated + (1 - self.alpha) *(0.9 * self.gradn + 0.1 * aggregated_current)

        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(self.gradn)
        self.server.apply_gradient()

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        self.parallel_call(lambda worker: worker.train_epoch_start())

        progress = 0
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                self._run_pre_batch_hooks(epoch, batch_idx)
                results = self.parallel_get(lambda w: w.compute_gradient())
                self.aggregation_and_update()

                progress += sum(res["length"] for res in results)
                if batch_idx % self.log_interval == 0:
                    self.log_train(progress, batch_idx, epoch, results)
                self._run_post_batch_hooks(epoch, batch_idx)
            except StopIteration:
                break

    # ---------------------------------------------------------------------------- #
    #                                    Utility                                   #
    # ---------------------------------------------------------------------------- #

    def add_worker(self, worker: TorchWorker):
        worker.add_metrics(self.metrics)
        self.workers.append(worker)
        self.debug_logger.info(f"=> Add worker {worker}")

    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def cache_random_state(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def parallel_call(self, f: Callable[[TorchWorker], None]) -> None:
        for w in self.workers:
            self.cache_random_state()
            f(w)
            self.restore_random_state()

    def parallel_get(self, f: Callable[[TorchWorker], Any]) -> list:
        results = []
        for w in self.workers:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results

    def _run_pre_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.pre_batch_hooks]

    def _run_post_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.post_batch_hooks]

    # ---------------------------------------------------------------------------- #
    #                                Log information                               #
    # ---------------------------------------------------------------------------- #

    def __str__(self):
        return (
            "ParallelTrainer("
            f"aggregator={self.aggregator}, "
            f"max_batches_per_epoch={self.max_batches_per_epoch}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["length"] for res in results) / length,
        }

        for metric_name in self.metrics:
            r[metric_name] = (
                sum(res["metrics"][metric_name] * res["length"] for res in results)
                / length
            )

        # Output to console
        total = len(self.workers[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)


class DistributedEvaluator(DistributedSimulatorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        log_identifier_type="validation",
    ):
        super().__init__(metrics, use_cuda, debug)
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.log_identifier_type = log_identifier_type

    def __str__(self):
        return (
            "DistributedEvaluator("
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": {"type": self.log_identifier_type},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output, target).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )
