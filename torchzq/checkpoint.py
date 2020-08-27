import torch
from pathlib import Path


class Checkpoint:
    def __init__(
        self, root, model, optimizer=None, amp=None, continue_=False, last_epoch=None,
    ):
        self.root = root
        self.model = model
        self.optimizer = optimizer
        self.amp = amp
        self.continue_ = continue_

        if last_epoch is None:
            self.last_epoch = self.get_last_epoch()
        else:
            self.last_epoch = last_epoch

    def get_last_epoch(self):
        paths = list(self.root.glob("*.pth"))

        if self.model.training:
            if paths and not self.continue_:
                print("Some checkpoints exist and continue not set, skip training.")
                exit()
        elif not paths:
            print("No checkpoint exists, no way to test.")
            exit()

        if paths:
            last_epoch = int(max(paths, key=lambda p: int(p.stem)).stem)
        else:
            last_epoch = -1

        return last_epoch

    def load(self):
        if self.last_epoch >= 0:
            path = self.root / f"{self.last_epoch}.pth"
            state_dict = torch.load(path)
            if "model" not in state_dict:
                # for backward compatiblility
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict["model"])
                # during testing, optimizer is None
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(state_dict["optimizer"])
                # without fp16, amp is None
                if self.amp is not None:
                    self.amp.load_state_dict(state_dict["amp"])
            print(f"{path} loaded.")

    def save(self, epoch):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{epoch}.pth"
        state_dict = dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            amp=self.amp.state_dict() if self.amp is not None else {},
        )
        torch.save(state_dict, path)
        print(f"{path} saved.")
