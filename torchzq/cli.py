import re
import yaml
import argparse
from pathlib import Path


class empty_value:
    pass


def load_yaml(path, command):
    data = {}
    with open(path, "r") as f:
        # remove the blank at the end of lines
        # to ensure yaml print multilines correctly
        lines = f.read().splitlines()
        data = "\n".join([l.rstrip() for l in lines])
        data = yaml.load(data, Loader=yaml.FullLoader)
    data = data or {}

    if command in data:
        data = {**data, **data[command]}

    # except for command, only support first level
    for key in list(data.keys()):
        if type(data[key]) is dict:
            del data[key]

    return data


def update_tree(a, b):
    for k in b:
        if k in a and type(a[k]) is dict and type(b[k]) is dict:
            update_tree(a[k], b[k])
        else:
            a[k] = b[k]


def parse_yaml(path, command):
    data = load_yaml(path, command)
    result = {}
    defaults = data.get("default", [])
    if type(defaults) is not list:
        defaults = [defaults]
    for path in defaults:
        update_tree(result, parse_yaml(path, command))
    update_tree(result, data)
    return result


class ConfigParser:
    def __init__(self, path):
        self.path = Path(path)

    def parse_manual_options(self, manual_options):
        parsed = []
        for s in manual_options:
            if s.startswith("--"):
                parsed.append([s.lstrip("-")])
            else:
                parsed[-1].append(s)
        for i, s in enumerate(parsed):
            if len(s) > 2:
                parsed[i] = [s[0], s[1:]]
        return parsed

    @staticmethod
    def parse_key(key):
        return "--" + key.replace("_", "-").lstrip("-")

    @staticmethod
    def parse_value(value):
        if type(value) is list:
            return " ".join(map(ConfigParser.parse_value, value))
        elif " " in str(value) or "(" in str(value):
            value = f'"{value}"'
        return str(value)

    def parse_option(self, k, v=empty_value):
        if k == "name":
            groups = re.findall("\$from\((.+?)\)", v)
            if groups:
                path = Path(groups[0])
                if path not in self.path.parents:
                    raise ValueError(
                        "Name root should be the parent of the configuration file!"
                    )
                rpath = self.path.relative_to(path)
                v = rpath.with_suffix("")
        return f"{self.parse_key(k)} {'' if v is empty_value else self.parse_value(v)}"

    def parse_cmd(self, command, manual_arguments, manual_options):
        data = parse_yaml(self.path, command)

        if "default" in data:
            del data["default"]

        if "runner" not in data:
            raise ValueError(
                "No runner detected, please specify a runner in the config file."
            )

        runner = data["runner"]
        del data["runner"]

        options = []
        for kv in list(data.items()) + self.parse_manual_options(manual_options):
            options.append(self.parse_option(*kv))
        return f"{runner} {command} {' '.join(manual_arguments)} " + " ".join(options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    parser.add_argument("command")
    args, manual_inputs = parser.parse_known_args()

    try:
        manual_arguments, manual_options = " ".join(manual_inputs).split("--", 1)
        manual_arguments = manual_arguments.split()
        manual_options = ("--" + manual_options).split()
    except:
        manual_arguments = manual_inputs
        manual_options = []

    print(
        ConfigParser(args.yaml).parse_cmd(
            args.command,
            manual_arguments,
            manual_options,
        )
    )
