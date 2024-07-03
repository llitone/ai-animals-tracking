import os

import random
import shutil
import yaml
import hashlib


class DatasetGenerator(object):
    def __init__(self, path):
        self.__class_number = 0
        self.__hash = hashlib.sha256(str(random.random()).encode())
        self.path = path
        self.classes = {}

        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(os.path.join(path, 'images')):
            os.mkdir(os.path.join(path, 'images'))
        if not os.path.isdir(os.path.join(path, 'labels')):
            os.mkdir(os.path.join(path, 'labels'))
        if not os.path.isdir(os.path.join(path, 'images', 'train')):
            os.mkdir(os.path.join(path, 'images', 'train'))
        if not os.path.isdir(os.path.join(path, 'labels', 'train')):
            os.mkdir(os.path.join(path, 'labels', 'train'))
        if not os.path.isdir(os.path.join(path, 'images', 'val')):
            os.mkdir(os.path.join(path, 'images', 'val'))
        if not os.path.isdir(os.path.join(path, 'labels', 'val')):
            os.mkdir(os.path.join(path, 'labels', 'val'))
        if not os.path.isdir(os.path.join(path, 'images', 'test')):
            os.mkdir(os.path.join(path, 'images', 'test'))
        if not os.path.isdir(os.path.join(path, 'labels', 'test')):
            os.mkdir(os.path.join(path, 'labels', 'test'))

    def add_class(self, class_name):
        self.classes[self.__class_number] = class_name
        self.__class_number += 1

    def add_image(self, image_path, image_classes, *, image_type: str = "train"):
        filename = image_path.split('/')[-1].split('.')
        self.__hash.update("".join(filename[:-1]).encode())
        new_image_path = os.path.join(self.path, "images", image_type, f"{self.__hash.hexdigest()}.{filename[-1]}")
        shutil.copyfile(image_path, new_image_path)

        with open(os.path.join(self.path, 'labels', image_type, f"{self.__hash.hexdigest()}.txt"), '+w') as f:
            f.writelines(image_classes)

    def __call__(self):
        _, _, files = next(os.walk(os.path.join(self.path, 'images', 'test')))
        test_files_count = len(files)
        yaml_data = {
            "path": self.path,
            "train": "images/train",
            "val": "images/val",
            "test": "" if test_files_count == 0 else "images/test",
            "names": self.classes
        }
        with open(os.path.join(self.path, "data.yaml"), '+w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
