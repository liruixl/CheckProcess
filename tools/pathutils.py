import re


def splict_path(path):
    return re.split('\\\\+|/+', path)[-1]

