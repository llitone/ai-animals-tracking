import numpy as np
import torch

from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path

try:
    from yolov5.utils.augmentations import letterbox
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadScreenshots, LoadImages
    from yolov5.utils.general import (check_file, increment_path, check_img_size, Profile,
                                      non_max_suppression, scale_boxes, xyxy2xywh)
    from yolov5.utils.torch_utils import select_device
except ModuleNotFoundError:
    import sys

    sys.path.append("./yolov5/")

    from yolov5.utils.augmentations import letterbox
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadScreenshots, LoadImages
    from yolov5.utils.general import (check_file, increment_path, check_img_size, Profile,
                                      non_max_suppression, scale_boxes, xyxy2xywh)
    from yolov5.utils.torch_utils import select_device


class Output(object):
    XYXY: list[float]
    XYWH: list[float]
    PROB: float
    LABEL: str
    FRAME: torch.Tensor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.FRAME is None:
            return f"Output({self.XYXY}, {self.XYWH}, {self.PROB}, {self.LABEL})"
        return f"Output({self.XYXY}, {self.XYWH}, {self.PROB}, {self.LABEL}, {self.FRAME.shape})"

    def __str__(self):
        if self.FRAME is None:
            return f"Output(\n\t{self.XYXY}, \n\t{self.XYWH}, \n\t{self.PROB}, \n\t{self.LABEL})"
        return f"Output(\n\t{self.XYXY}, \n\t{self.XYWH}, \n\t{self.PROB}, \n\t{self.LABEL}, \n\t{self.FRAME.shape})"


class Dataset(object):
    def __init__(self, raw_frame):
        self.raw_frame = raw_frame
        self.mode = "image"
        self.imgsz = (640, 640)
        self.stride = 0

    def __iter__(self):
        im = letterbox(self.raw_frame, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        return iter([("./result.jpg", im, im, "", "")])

    def __len__(self):
        return 1

    def __next__(self):
        im = letterbox(self.raw_frame, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        return "./result.jpg", im, im, "", ""


class Model(DetectMultiBackend):
    def __init__(self, weights, *args, **kwargs) -> None:
        super().__init__(*args, weights=weights, **kwargs)

    @staticmethod
    def intersection(user_box, true_box):
        x, y, w, h = user_box
        user_box = (x, y, x + w, y + h)

        x, y, w, h = true_box
        true_box = (x, y, x + w, y + h)

        x1 = max(user_box[0], true_box[0])
        y1 = max(user_box[1], true_box[1])
        x2 = min(user_box[2], true_box[2])
        y2 = min(user_box[3], true_box[3])

        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        return inter_area > 0

    def predict_one_sample(
            self,
            source="data/images",  # file/dir/URL/glob/screen/0(webcam),
            use_tqdm: bool = False,  # TODO
            scale: bool = False,
            return_frame=False,  # xyxy, xywh, frame
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            project="runs/detect",  # save results to project/name
            name="exp",  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            hide_conf=False,  # hide confidences
            vid_stride=1,  # video frame-rate stride
    ) -> list[Output]:  # TODO: frames array
        raw_frame = None
        if isinstance(source, list | np.ndarray):
            if isinstance(source[0], np.ndarray | list | int):
                raw_frame = source.copy()
        source = str(source)
        save_img = not nosave and not source.endswith(".txt")  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower().startswith("screen")
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        stride, names, pt = self.stride, self.names, self.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if raw_frame is not None:
            dataset = Dataset(raw_frame)
            dataset.imgsz = imgsz
            dataset.stride = stride
        else:
            if webcam:
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.warmup(imgsz=(1 if pt or self.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        outputs = []
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if self.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if self.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = self(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, self(image, augment=augment, visualize=visualize).unsqueeze(0)),
                                             dim=0)
                    pred = [pred, None]
                else:
                    pred = self(im, augment=augment, visualize=visualize)

            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            out = []
            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}")
                s += "%gx%g " % im.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                annotator = Annotator(im0, line_width=3, example=str(names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        output = Output()
                        output.XYXY = list(map(lambda x: x.item(), xyxy))
                        output.XYWH = xywh
                        output.LABEL = label
                        output.PROB = confidence

                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")
                        if save_img or return_frame:
                            c = int(cls)
                            label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        out.append(output)

                im0 = annotator.result()
                if return_frame:
                    for outp in out:
                        outp.FRAME = im0.copy()
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)
            outputs.append(out)
        return outputs[0]


if __name__ == '__main__':
    import cv2

    model = Model("./yolov5n.pt")
    model.eval()

    video = cv2.VideoCapture(0)
    cv2.startWindowThread()
    # cv2.namedWindow("Frame")
    # cv2.namedWindow("BBFrame")
    # frame = cv2.imread("test.jpg")
    # cv2.imshow(f"Frame", frame)

    cv2.waitKey(10000)
    key = 0
    while key != 27:
        ret, frame = video.read()
        if ret:
            cv2.imshow("Frame", cv2.resize(frame, (300, 300)))
            image = frame.copy()
            preds = model.predict_one_sample(image, return_frame=True)
            if len(preds) == 0:
                key = cv2.waitKey(100)
                continue

        key = cv2.waitKey(100)
    video.release()
    cv2.destroyAllWindows()
