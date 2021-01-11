import torch
from pathlib import Path


def load_state_dict_safe(model, state_dict, strict):
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state_dict = model.state_dict()
        provided = set(state_dict)
        required = set(model_state_dict)
        agreed = provided & required
        for k in list(agreed):
            if model_state_dict[k].shape != state_dict[k].shape:
                agreed.remove(k)
        state_dict = {k: state_dict[k] for k in agreed}
        print("Provided but not required keys: ")
        print(provided - required)
        print("Required but not provided keys: ")
        print(required - provided)
        model.load_state_dict(state_dict, strict=False)
    return model


class Saver:
    def __init__(self, root, strict=True):
        self.root = root
        self.strict = strict
        self.cache = {}

    @property
    def empty(self):
        return self.latest_epoch is None

    @property
    def latest_epoch(self):
        if not hasattr(self, "_latest_epoch") is None:
            paths = list(self.root.glob("*.pth"))
            if paths:
                self._latest_epoch = int(max(paths, key=lambda p: int(p.stem)).stem)
            else:
                self._latest_epoch = None
        return self._latest_epoch

    def read_state_dict(self, epoch, cache=False):
        path = self.root / f"{epoch}.pth"
        if path in self.cache:
            state_dict = self.cache[path]
        else:
            state_dict = torch.load(path, "cpu")
            if cache:
                self.cache[path] = state_dict
        return state_dict

    def load(
        self,
        model=None,
        optimizer=None,
        scaler=None,
        epoch=None,
        cache=False,
    ):
        epoch = self.latest_epoch if epoch is None else epoch

        if epoch is None:
            return

        state_dict = self.read_state_dict(epoch, cache)

        if model is not None:
            load_state_dict_safe(model, state_dict["model"], self.strict)
            model.epoch = epoch
            model.iteration = state_dict.get("iteration", 0)
            print(f"==> Model at epoch {epoch} loaded.")

        if optimizer is not None:
            try:
                optimizer.load_state_dict(state_dict["optimizer"])
                print(f"==> Optimizer at epoch {epoch} loaded.")
            except Exception as e:
                print(e)
                print("Warning: loading optimizer state dict failed.")

        if scaler is not None:
            try:
                scaler.load_state_dict(state_dict["scaler"])
                print(f"==> Scaler at epoch {epoch} loaded.")
            except Exception as e:
                print(e)
                print("Warning: loading scaler state dict failed.")

    def save(self, model, optimizer=None, scaler=None):
        self.root.mkdir(parents=True, exist_ok=True)
        state_dict = dict(
            iteration=model.iteration,
            epoch=model.epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else None,
            scaler=scaler.state_dict() if scaler else None,
        )
        path = self.root / f"{model.epoch}.pth"
        torch.save(state_dict, path)
        print(f"{path} saved.")
