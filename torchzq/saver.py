import torch
from pathlib import Path


def load_state_dict(module, state_dict, strict):
    if strict:
        module.load_state_dict(state_dict, strict=True)
    else:
        provided = set(state_dict)
        required = set(module.state_dict())
        agreed = provided & required
        state_dict = {k: state_dict[k] for k in agreed}
        print("Provided but not required keys: ")
        print(provided - required)
        print("Required but not provided keys: ")
        print(required - provided)
        module.load_state_dict(state_dict, strict=False)
    return module


class Saver:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.last_epoch = self.get_last_epoch()

    @property
    def is_empty(self):
        return self.last_epoch == 0

    def get_last_epoch(self):
        paths = list(self.root.glob("*.pth"))
        if paths:
            last_epoch = int(max(paths, key=lambda p: int(p.stem)).stem)
        else:
            last_epoch = 0
        return last_epoch

    def load(self, model, optimizer=None, epoch=None, strict=True):
        epoch = self.last_epoch if epoch is None else epoch
        iteration = 0

        if epoch > 0:
            path = self.root / f"{epoch}.pth"

            if path not in self.cache:
                self.cache[path] = torch.load(path)

            state_dict = self.cache[path]

            load_state_dict(model, state_dict["model"], strict)

            iteration = state_dict.get("iteration", 0)

            # during testing, optimizer is None
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(state_dict["optimizer"])
                except Exception as e:
                    print(e)
                    # allow not loading the optimizer
                    print("Warning: loading optimizer failed.")

            # without fp16, amp is None
            if model.amp is not None:
                model.amp.load_state_dict(state_dict["amp"])

            print(f"{path} loaded.")

        model.epoch = epoch
        model.iteration = iteration

    def save(self, model, optimizer=None):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{model.epoch}.pth"
        state_dict = dict(
            iteration=model.iteration,
            epoch=model.epoch,
            model=model.state_dict() if model else None,
            optimizer=optimizer.state_dict() if optimizer else None,
            amp=model.amp.state_dict() if model.amp else None,
        )
        torch.save(state_dict, path)
        print(f"{path} saved.")
