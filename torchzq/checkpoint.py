import torch
from pathlib import Path


def prepare(model, ckpt_dir, continue_=False, last_epoch=None):
    """Prepare a model for checkpoints.
    """
    ckpts = list(ckpt_dir.glob("*.pth"))

    if model.training:
        if ckpts and not continue_:
            print("Some ckpts exists and continue not set, skip training.")
            exit()
    elif not ckpts:
        print("No ckpt exists, no way to test.")
        exit()

    if last_epoch is not None:
        ckpts = [Path(ckpt_dir, f"{last_epoch}.pth")]

    if ckpts:
        ckpt = max(ckpts, key=lambda p: int(p.stem))
        last_epoch = int(ckpt.stem)
        try:
            model.load_state_dict(torch.load(ckpt))
            print(f"{ckpt} loaded.")
        except Exception as e:
            print(e)
            print(f"{ckpt} loading failed, start from scratch.")
    else:
        last_epoch = -1

    model.last_epoch = last_epoch

    if model.training:

        def save(epoch):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = Path(ckpt_dir, f"{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"{ckpt} saved.")

        model.save = save

    return model
