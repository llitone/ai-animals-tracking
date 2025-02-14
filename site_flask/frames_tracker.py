from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm.auto import tqdm

from model import Model


class AnimalImage(object):
    def __init__(self, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.datetime = ""

        image = Image.open(self.filename)
        exif_data = image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTime":
                    self.datetime = value
                    break
        if self.datetime != "":
            self.datetime = datetime.strptime(self.datetime, "%Y:%m:%d %H:%M:%S")
        self.objects = []

    def __repr__(self):
        return f"AnimalImage({self.datetime})"


class FramesTracker(Model):
    def __init__(self, path_to_folder, detector_path: str, classifier_path: str, le_path: str, files):
        super().__init__(detector_path, classifier_path, le_path)
        self.path_to_folder = path_to_folder
        self.files = list(map(AnimalImage, files))
        self.current_animals = {}
        self.all_animals = []

    def sort(self, *, reverse=False):
        self.files.sort(key=lambda x: x.datetime, reverse=reverse)


def predict(folder_path, files):
    tracker = FramesTracker(
        folder_path + "/*.JPG",
        "yolov8n-2.pt",
        "last_animal_model-4.pt",
        "le.bf",
        files
    )
    tracker.sort()

    for i in tqdm(tracker.files):
        i.objects = tracker.predict(i)

    for i in range(len(tracker.files)):
        time_has_passed = defaultdict(lambda: 0)
        if i != 0:
            for j in tracker.files[i].objects:
                if j.cls in tracker.current_animals:
                    time_has_passed[j.cls] = (
                            tracker.files[i].datetime - tracker.current_animals[j.cls]["last_seen"]
                    ).seconds
        for j in tracker.files[i].objects:
            if time_has_passed[j.cls] > 30 * 60:
                tracker.all_animals.append(deepcopy(tracker.current_animals[j.cls]))
                del tracker.current_animals[j.cls]
            if j.cls != "Empty":
                if j.cls not in tracker.current_animals:
                    tracker.current_animals[j.cls] = {
                        "first_seen": tracker.files[i].datetime,
                        "last_seen": tracker.files[i].datetime,
                        "count": 0,
                        "cls": j.cls,
                        "filename": tracker.files[i].filename
                    }
                tracker.current_animals[j.cls]["count"] = max(
                    tracker.current_animals[j.cls]["count"],
                    len([1 for animal in tracker.files[i].objects if animal.cls == j.cls])
                )
                tracker.current_animals[j.cls]["last_seen"] = tracker.files[i].datetime

    for i in tracker.current_animals:
        tracker.all_animals.append(deepcopy(tracker.current_animals[i]))
    # print(tracker.all_animals)
    return tracker.all_animals
