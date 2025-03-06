import json


def load_shader_source(path):
    with open(path, 'r') as file:
        return file.read()

def load_trajectory(path):
    with open(path, 'r') as file:
        return json.load(file)