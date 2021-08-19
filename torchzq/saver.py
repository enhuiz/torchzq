import torch
import torch.nn as nn
from pathlib import Path

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

    def __str__(self):
        return f"epoch={self.epoch}-step={self.step}"

    def __repr__(self):
        return f"State(epoch={self.epoch}, step={self.step})"


class Saver:
    def __init__(self, root, strict: bool = False):
        self.root = Path(root)
        self.strict = strict
        self.buffered = dict()
        self._preload()

    def _preload(self):
        for folder in self.root.glob("*"):
            namespace = folder.name
            path = self.find_latest_ckpt_path(namespace)
            if path and path.exists():
                data = torch.load(path, "cpu")
                self.buffered[namespace] = data
                # for compatibility
                if "state" not in data:
                    data["state"] = str(State.from_string(path.stem))

    def _get_ckpt_path(self, namespace, stem):
        return (self.root / namespace / stem).with_suffix(".ckpt")

    def find_ckpt_paths(self, namespace="default"):
        return list((self.root / namespace).glob("*.ckpt"))

    def find_latest_ckpt_path(self, namespace):
        paths = self.find_ckpt_paths(namespace)
        return max(paths, key=lambda p: State.parse(p.stem)[-1], default=None)

    def load(
        self,
        model=None,
        optimizers=[],
        scaler=None,
        namespace="default",
        path=None,
    ):
        if path is None:
            if namespace not in self.buffered:
                msg = f"No checkpoint found in {self.root / namespace}."
                if namespace is not "default":
                    # strict checking for non-default namespace
                    raise FileNotFoundError(msg)
                if model is not None:
                    model.state = State()
                print(msg, "Not loading.")
                return

            data = self.buffered[namespace]
        else:
            data = torch.load(path, "cpu")

        if model is not None:
            if self.strict:
                model.load_state_dict(data["model"])
            else:
                unsafe_load_state_dict(model, data["model"])
            model.state = State.from_string(data["state"])
            print(f"{self._get_ckpt_path(namespace, str(model.state))} loaded.")

        state_dicts = data.get("optimizers", [])

        for i, (optimizer, state_dict) in enumerate(zip(optimizers, state_dicts)):
            try:
                optimizer.load_state_dict(state_dict)
                # sanity check
                optimizer.zero_grad()
                optimizer.step()
                print(f"Optimizer loaded.")
            except Exception as e:
                print(e)
                print(f"Warning: fail to load state dict for optimizer {i}.")

        if scaler is not None:
            try:
                scaler.load_state_dict(data["scaler"])
                print(f"Scaler loaded.")
            except Exception as e:
                print(e)
                print("Warning: loading scaler state dict failed.")

    def buffer(self, model, optimizers=[], scaler=None, namespace="default"):
        state = model.state
        del model.state
        self.buffered[namespace] = dict(
            model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],
            scaler=scaler.state_dict() if scaler else None,
            state=str(state),
        )
        model.state = state

    def dump(self, namespace="default"):
        data = self.buffered[namespace]
        if State.parse_step(data["state"]) > 0:
            path = self._get_ckpt_path(namespace, data["state"])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, path)
            print(f"{path} dumped.")
        del self.buffered[namespace]

    def dump_all(self):
        for namespace in list(self.buffered):
            self.dump(namespace)
