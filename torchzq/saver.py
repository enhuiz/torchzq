import torch
from queue import Queue
from pathlib import Path
from functools import lru_cache
from collections import defaultdict


def load_state_dict_lenient(model, state_dict):
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


@lru_cache
def cached_load_ckpt(path):
    return torch.load(path, "cpu")


class Saver:
    def __init__(self, root: Path, strict: bool = False, buffer_size=1):
        self.root = root
        self.strict = strict
        self.buffer_dict = defaultdict(lambda: Queue(buffer_size))

    def get_ckpts(self, namespace=None):
        return list((self.root / (namespace or "")).glob("*.ckpt"))

    def get_latest_ckpt(self, namespace):
        return max(
            self.get_ckpts(namespace),
            key=lambda ckpt: self.parse(ckpt)[1],
            default=None,
        )

    def get_path(self, model, namespace=None):
        state = model.state
        name = f"epoch={state.current_epoch}-step={state.global_step}.ckpt"
        return self.root / (namespace or "") / name

    def parse(self, path):
        epoch, step = map(lambda kv: int(kv.split("=")[1]), path.stem.split("-"))
        return epoch, step

    def load(self, ckpt, model=None, optimizers=[], scaler=None):
        if ckpt is None:
            return

        state_dict = cached_load_ckpt(ckpt)

        if "state._current_epoch" not in state_dict:
            # for compatibility
            i2t = lambda i: torch.tensor(i)
            epoch, step = self.parse(ckpt)
            state_dict["model"]["state._current_epoch"] = i2t(epoch)
            state_dict["model"]["state._global_step"] = i2t(step)

        if model is not None:
            if self.strict:
                model.load_state_dict(state_dict["model"])
            else:
                load_state_dict_lenient(model, state_dict["model"])
            state = model.state
            print(
                f"Model epoch={state.current_epoch}",
                f"step={state.global_step} loaded.",
            )

        for i, (optimizer, optimizer_state_dict) in enumerate(
            zip(optimizers, state_dict.get("optimizers", []))
        ):
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                # sanity check
                optimizer.zero_grad()
                optimizer.step()
                print(f"Optimizer loaded.")
            except Exception as e:
                print(e)
                print(f"Warning: fail to load state dict for optimizer {i}.")

        if scaler is not None:
            try:
                scaler.load_state_dict(state_dict["scaler"])
                print(f"Scaler loaded.")
            except Exception as e:
                print(e)
                print("Warning: loading scaler state dict failed.")

    def buffer(self, model, optimizers=[], scaler=None, namespace=None):
        path = self.get_path(model, namespace)
        data = dict(
            model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],
            scaler=scaler.state_dict() if scaler else None,
        )
        if self.buffer_dict[namespace].full():
            self.buffer_dict[namespace].get()
        self.buffer_dict[namespace].put([path, data])

    def dump(self, namespace=None):
        while not self.buffer_dict[namespace].empty():
            path, data = self.buffer_dict[namespace].get()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, path)
            print(f"{path} saved.")

    def dump_all(self):
        for namespace in self.buffer_dict.keys():
            self.dump(namespace)

    def save(self, model, optimizers=[], scaler=None):
        self.buffer(model, optimizers, scaler)
        self.dump()
