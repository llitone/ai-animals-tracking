import cv2
import numpy as np
import torch
import pickle

from torch import nn
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
from torchvision.transforms.functional import normalize
from itertools import repeat
from pathlib import Path


class Prediction(object):
    def __init__(self, name, cls, prob, bbox):
        super().__init__()
        self.name = name
        self.cls = cls
        self.prob = prob
        self.bbox = bbox

    def __repr__(self):
        return f"Prediction({self.name}, {self.cls})"


class Model(object):
    CLASSES = ["Badger", "Bear", "Bison", "Cat", "Dog",
               "Empty", "Fox", "Goral", "Hare", "Lynx",
               "Marten", "Moose", "Mountain_Goat",
               "Musk_Deer", "Racoon_Dog", "Red_Deer",
               "Roe_Deer", "Snow_Leopard", "Squirrel",
               "Tiger", "Wolf", "Wolverine"]

    def __init__(self, detector_path: str, classifier_path: str, le_path: str):
        self.detector = YOLO(detector_path)
        self.classifier = models.vgg19(pretrained=False)
        # for param in self.classifier.parameters():
        #     param.requires_grad = False
        self.classifier.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 19)
        )
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
        self.classifier.eval()

        with open(le_path, "rb") as file:
            self.le: LabelEncoder = pickle.load(file)

    def predict(self, frame):
        if not isinstance(frame, (np.ndarray, str)):
            frame = frame.filename
        # print(frame)
        detections = self.detector.predict(frame, verbose=False)
        classes = []
        croped_frames = self.extract_crops(detections)
        for (img_name, batch_images_cls) in croped_frames.items():
            logits = self.classifier(batch_images_cls.to("cpu"))
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top_p, top_class_idx = probabilities.topk(1, dim=1)

            top_p = top_p.cpu().detach().numpy().ravel()
            top_class_idx = top_class_idx.cpu().numpy().ravel()
            # print(top_class_idx)
            class_names = self.le.inverse_transform(top_class_idx)

            classes.extend(
                [
                    Prediction(name, cls, prob, bbox.xyxy)
                    for name, cls, prob, bbox in
                    zip(repeat(img_name, len(class_names)), class_names, top_p, detections[0].boxes)
                ]
            )

        return classes

    @staticmethod
    def extract_crops(results: list) -> dict[str, torch.Tensor]:
        dict_crops = {}
        for res_per_img in results:
            if len(res_per_img) > 0:
                crops_per_img = []
                for box in res_per_img.boxes:
                    x0, y0, x1, y1 = box.xyxy.cpu().numpy().ravel().astype(np.int32)
                    crop = res_per_img.orig_img[y0: y1, x0: x1]

                    # Do squared crop
                    # crop = letterbox(img=crop, new_shape=config.imgsz, color=(0, 0, 0))
                    crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_LINEAR)
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('crop', crop)
                    # cv2.waitKey(111111110)
                    # Convert Array crop to Torch tensor with [batch, channels, height, width] dimensions
                    crop = torch.from_numpy(crop.transpose(2, 0, 1)) / 255
                    crop = crop.unsqueeze(0)
                    crop = normalize(crop.float(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    crops_per_img.append(crop)

                dict_crops[Path(res_per_img.path).name] = torch.cat(crops_per_img)  # if len(crops_per_img) else None
        return dict_crops


