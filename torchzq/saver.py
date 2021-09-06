import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional

from .utils import ItemProperty


def unsafe_load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    provided = set(state_dict)
    required = set(model_state_dict)
    agreed = provided & required
    for k in list(agreed):
        if model_state_dict[k].shape != state_dict[k].shape:
            agreed.remove(k)
            provided.remove(k)
    state_dict = {k: state_dict[k] for k in agreed}
    if diff := provided - required:
        print("Provided but not required keys: ")
        print(diff)
    if diff := required - provided:
        print("Required but not provided keys: ")
        print(required - provided)
    model.load_state_dict(state_dict, strict=False)
    return model


class State(nn.Module):
    """
    The state of the loop engine.
    """

    epoch = ItemProperty()
    step = ItemProperty()

    def __init__(self, epoch=0, step=0):
        super().__init__()
        self.register_buffer("_step", torch.ones([]).long() * step)
        self.register_buffer("_epoch", torch.ones([]).long() * epoch)

    @staticmethod
    def parse(s):
        """
        Returns:
            epoch, step
        """
        return [int(kv.split("=")[1]) for kv in s.split("-")]

    @classmethod
    def from_string(cls, s):
        epoch, step = cls.parse(s)
        return cls(epoch=epoch, step=step)

    @classmethod
    def parse_epoch(cls, s):
        return cls.parse(s)[0]

    @classmethod
    def parse_step(cls, s):
        return cls.parse(s)[1]

    @classmethod
    def from_state_dict(cls, state_dict):
        if isinstance(state_dict, str):
            # backward compatibility
            state = cls.from_string(state_dict)
        else:
            state = cls()
            if state_dict is not None:
                state.load_state_dict(state_dict)
        return state

    def to_path(self, root):
        return (root / str(self)).with_suffix(".ckpt")

    def __str__(self):
        return f"epoch={self.epoch}-step={self.step}"

    def __repr__(self):
        return f"State(epoch={self.epoch}, step={self.step})"


@dataclass(eq=False, frozen=True)
class Checkpoint:
    path: Optional[Path] = None
    state: Optional[OrderedDict] = None
    model: Optional[OrderedDict] = None
    optimizers: Optional[list[OrderedDict]] = None
    scaler: Optional[OrderedDict] = None
    metrics: Optional[OrderedDict] = None

    @classmethod
    def from_path(cls, path):
        state_dict = torch.load(path, "cpu")
        return cls(
            path=path,
            state=state_dict["state"],
            model=state_dict["model"],
            optimizers=state_dict["optimizers"],
            scaler=state_dict["scaler"],
            # use get for backward compatibility
            metrics=state_dict.get("metrics", None),
        )

    def info(self, msg):
        print(f"ckpt@{self.path}: {msg}")

    def load_state(self):
        return State.from_state_dict(self.state)

    def load_metrics(self, metrics):
        if self.metrics is not None:
            metrics.load_state_dict(self.metrics)
            self.info("metrics loaded.")

        return metrics

    def load_model(self, model, strict=False):
        if self.model is None:
            return

        if strict:
            model.load_state_dict(self.model)
        else:
            unsafe_load_state_dict(model, self.model)

        self.info("model loaded.")

        return model

    def load_optimizers(self, optimizers):
        if self.optimizers is None:
            return

        for i, (optimizer, state_dict) in enumerate(zip(optimizers, self.optimizers)):
            try:
                optimizer.load_state_dict(state_dict)
                # sanity check
                optimizer.zero_grad()
                optimizer.step()
                self.info(f"optimizer {i} loaded.")
            except Exception as e:
                self.info(f"failed to load state dict for optimizer {i}, skip.")
                self.info(e)

        return optimizers

    def load_scaler(self, scaler):
        if self.scaler is None:
            return

        try:
            scaler.load_state_dict(self.scaler)
            self.info("scaler loaded.")
        except Exception as e:
            self.info("failed to load scaler state dict.")
            self.info(e)

        return scaler

    def dump(self):
        if self.path is None:
            raise ValueError(f"{self} is not dumpable.")

        if self.path.exists():
            # skip dump to avoid overwriting existing checkpoint.
            return

        state = State.from_state_dict(self.state)

        if state.step > 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), self.path)
            self.info(f"dumped.")

    def state_dict(self):
        return dict(
            state=self.state,
            model=self.model,
            optimizers=self.optimizers,
            scaler=self.scaler,
            metrics=self.metrics,
        )

    def __eq__(self, other):
        return self.path == other.path


class Saver(dict):
    """
    Saver is a dict maps namespace to checkpoint
    """

    def __init__(self, root, strict: bool = False):
        self.root = Path(root)
        self.strict = strict
        self._preload()

    def _preload(self):
        for folder in self.root.glob("*"):
            namespace = folder.name
            path = self.get_latest_ckpt_path(namespace)
            if path and path.exists():
                self[namespace] = Checkpoint.from_path(path)

    def list_ckpt_paths(self, namespace="default"):
        return list((self.root / namespace).glob("*.ckpt"))

    def get_latest_ckpt_path(self, namespace):
        paths = self.list_ckpt_paths(namespace)
        return max(paths, key=lambda p: State.parse(p.stem)[-1], default=None)

    def buffer(
        self,
        state,
        model,
        optimizers=[],
        scaler=None,
        metrics=None,
        namespace="default",
    ):
        self[namespace] = Checkpoint(
            path=state.to_path(self.root / namespace),
            state=state.state_dict(),
            model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],
            scaler=scaler.state_dict() if scaler else None,
            metrics=metrics.state_dict() if metrics else None,
        )

    def dump(self, namespace="default"):
        self[namespace].dump()
        del self[namespace]

    def dump_all(self):
        for namespace in list(self):
            self.dump(namespace)
