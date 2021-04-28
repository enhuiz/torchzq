import contextlib
import torch
from pathlib import Path
from functools import lru_cache


def load_state_dict_non_strict(model, state_dict):
    model_state_dict = model.state_dict()
    provided = set(state_dict)
    required = set(model_state_dict)
    agreed = provided & required
    for k in list(agreed):
        if model_state_dict[k].shape != state_dict[k].shape:
            agreed.remove(k)
            provided.remove(k)
    state_dict = {k: state_dict[k] for k in agreed}
    if (diff := provided - required) :
        print("Provided but not required keys: ")
        print(diff)
    if (diff := required - provided) :
        print("Required but not provided keys: ")
        print(required - provided)
    model.load_state_dict(state_dict, strict=False)
    return model


@lru_cache
def cached_load_ckpt(path):
    return torch.load(path, "cpu")


class Buffer(dict):
    def __init__(self, to_path):
        self.to_path = to_path

    def __call__(self, model, optimizers=[], scaler=None):
        self[self.to_path(model)] = dict(
            iteration=model.iteration,
            epoch=model.epoch,
            model=model.state_dict(),
            optimizer=[optimizer.state_dict() for optimizer in optimizers],
            scaler=scaler.state_dict() if scaler else None,
        )

    def dump(self):
        for path, state_dict in self.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, path)
            print(f"{path} saved.")
        self.clear()


class Saver:
    def __init__(self, root: Path, strict: bool = True):
        self.root = root
        self.strict = strict
        self.buffer = Buffer(self.to_path)

    @property
    def ckpts(self):
        return list(self.root.glob("*.ckpt"))

    @property
    def empty(self):
        return len(self.ckpts) > 0

    @property
    def latest_ckpt(self):
        return max(self.ckpts, key=lambda ckpt: self.parse(ckpt)[1], default=None)

    def to_path(self, model):
        return self.root / f"epoch={model.epoch}-iteration={model.iteration}.ckpt"

    def parse(self, path):
        epoch, iteration = map(lambda kv: int(kv.split("=")[1]), path.stem.split("-"))
        return epoch, iteration

    def load(self, ckpt, model=None, optimizers=[], scaler=None, epoch=None):
        if ckpt is None:
            return

        state_dict = cached_load_ckpt(ckpt)

        if model is not None:
            if self.strict:
                load_state_dict_non_strict(model, state_dict["model"])
            else:
                model.load_state_dict(state_dict["model"])
            model.epoch, model.iteration = self.parse(ckpt)
            print(f"Model epoch={model.epoch} iteration={model.iteration} loaded.")

        for i, (optimizer, optimizer_state_dict) in enumerate(
            zip(
                optimizers,
                state_dict.get("optimizers", []),
            )
        ):
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                # sanity check
                optimizer.zero_grad()
                optimizer.step()
                print(f"Optimizer at epoch {epoch} loaded.")
            except Exception as e:
                print(e)
                print(f"Warning: fail to load state dict for optimizer {i}.")

        if scaler is not None:
            try:
                scaler.load_state_dict(state_dict["scaler"])
                print(f"Scaler at epoch {epoch} loaded.")
            except Exception as e:
                print(e)
                print("Warning: loading scaler state dict failed.")

    def save(self, model, optimizers=[], scaler=None):
        buffer = Buffer(self.to_path)
        buffer(model, optimizers, scaler)
        buffer.dump()
