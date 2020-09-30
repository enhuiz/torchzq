import argparse
import sys
import yaml


def load_yaml(path):
    data = {}
    with open(path, "r") as f:
        # remove the blank at the end of lines
        # to ensure yaml print multilines correctly
        lines = f.read().splitlines()
        data = "\n".join([l.rstrip() for l in lines])
        data = yaml.load(data, Loader=yaml.FullLoader)
    data = data or {}
    return data


def update_tree(a, b):
    for k in b:
        if k in a and type(a[k]) is dict and type(b[k]) is dict:
            update_tree(a[k], b[k])
        else:
            a[k] = b[k]


def parse_yaml(path):
    data = load_yaml(path)
    result = {}
    for path in data.get("default", []):
        result = parse_yaml(path)
    update_tree(result, data)
    return result


class ConfigParser:
    def __init__(self, path):
        self.path = path

    def parse_manual_options(self, manual_options):
        parsed = []
        for s in manual_options:
            if s.startswith("--"):
                parsed.append([s.lstrip("-")])
            else:
                parsed[-1].append(s)
        return parsed

    @staticmethod
    def normalize_key(key):
        return "--" + key.replace("_", "-").lstrip("-")

    def parse_cmd(self, command, manual_options=[]):
        data = parse_yaml(self.path)
        if command in data:
            data = {**data, **data[command]}

        if "default" in data:
            del data["default"]

        if "runner" not in data:
            raise ValueError(
                "No runner detected, please specify a runner in the config file."
            )

        runner = data["runner"]
        del data["runner"]

        options = []
        for k, v in list(data.items()):
            if type(v) is not dict:
                options.append(f'{self.normalize_key(k)} "{v}"')

        for kv in self.parse_manual_options(manual_options):
            try:
                k, v = kv
                options.append(f'{self.normalize_key(k)} "{v}"')
            except:
                (k,) = kv
                options.append(f"{self.normalize_key(k)}")

        return f"{runner} {command} " + " ".join(options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    parser.add_argument("command")
    args, manual_options = parser.parse_known_args()
    print(ConfigParser(args.yaml).parse_cmd(args.command, manual_options))
