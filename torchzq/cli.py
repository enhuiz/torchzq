import os
import argparse
from pathlib import Path
from zouqi.utils import load_yaml


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("command", type=str)
    args, manual_argv = parser.parse_known_args()

    data = load_yaml(args.yaml, args.command)

    if "runner" not in data:
        raise ValueError("Please specify a runner in your config file.")

    runner = data["runner"]
    del data["runner"]

    if "name" not in data and "--name" not in manual_argv:
        rpath = args.yaml.relative_to(".").with_suffix("")
        if str(rpath).startswith("..") or len(rpath.parts) < 2:
            raise ValueError(
                f"Fail to generate a proper name for {args.yaml}. "
                "Please make sure it is under a subdirectory of the current directory."
            )
        manual_argv.extend(["--name", str(Path(*rpath.parts[1:]))])

    for i in range(len(manual_argv)):
        if not manual_argv[i].startswith("--"):
            manual_argv[i] = f"'{manual_argv[i]}'"

    cmd = f"{runner} {args.command} --config {args.yaml} {' '.join(manual_argv)} --config-ignored runner"
    os.system(cmd)


if __name__ == "__main__":
    main()
