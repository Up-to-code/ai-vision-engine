"""Independent optional perception backends, loaded only when requested."""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import Person, TextRegion


class OptionalModules:
    def __init__(self, config, manager):
        self.config, self.manager = config, manager
        self.cache = {}

    def ocr(self, image):
        if "ocr" not in self.cache:
            import easyocr

            self.cache["ocr"] = easyocr.Reader(self.config.ocr_languages, gpu=False, verbose=False)
        return [
            TextRegion(text=text, confidence=float(score), polygon=poly)
            for poly, text, score in self.cache["ocr"].readtext(image)
            if score >= 0.35
        ]

    def pose(self, image):
        result = self.manager.predict(
            "models/yolo26n-pose.pt",
            image,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
        )
        people = []
        if result.keypoints is None:
            return people
        for box, joints in zip(
            result.boxes.xyxy.cpu().tolist(), result.keypoints.data.cpu().tolist()
        ):
            person = Person(bbox=tuple(box), keypoints=joints)
            # Static geometry can support an arm-raised estimate, never walking/running.
            for shoulder, wrist in ((5, 9), (6, 10)):
                if joints[shoulder][2] > 0.5 and joints[wrist][2] > 0.5:
                    if joints[wrist][1] < joints[shoulder][1]:
                        person.activity = "arm_raised_looking"
                        person.evidence = "Confident wrist above shoulder in this frame"
            people.append(person)
        return people

    def depth(self, image):
        if "depth" not in self.cache:
            from transformers import pipeline

            self.cache["depth"] = pipeline(
                "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu"
            )
        output = self.cache["depth"](Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        depth = output["predicted_depth"].detach().cpu().numpy().squeeze()
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    def segmentation(self, image, objects):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        if "segmentation" not in self.cache:
            self.cache["segmentation"] = (
                AutoImageProcessor.from_pretrained(name),
                AutoModelForSemanticSegmentation.from_pretrained(name).eval(),
            )
        processor, model = self.cache["segmentation"]
        inputs = processor(
            images=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
        probabilities = logits.softmax(dim=1)[0]
        confidence, labels = probabilities.max(dim=0)
        labels, confidence = labels.numpy(), confidence.numpy()
        # Cityscapes road=0 and sidewalk=1. Roads are uncertain, not safe walking claims.
        sidewalk = (labels == 1) & (confidence >= 0.6)
        road = labels == 0
        blocked = np.zeros(labels.shape, dtype=bool)
        for obj in objects:
            if obj.class_name in {"person", "car", "truck", "bus", "bicycle", "motorcycle"}:
                x1, y1, x2, y2 = map(int, obj.bbox)
                blocked[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
        colors = np.zeros_like(image)
        colors[road | (confidence < 0.6)] = (0, 200, 220)
        colors[sidewalk & ~blocked] = (60, 210, 80)
        colors[blocked] = (40, 40, 230)
        overlay = cv2.addWeighted(image, 0.65, colors, 0.35, 0)
        return overlay, {
            "road_visible": bool(road.any()),
            "sidewalk_visible": bool(sidewalk.any()),
            "candidate_walkable_fraction": float((sidewalk & ~blocked).mean()),
            "warning": "Experimental image regions, no metric safety or trajectory",
        }

    def face(self, image):
        import mediapipe as mp

        if "face" not in self.cache:
            path = Path(self.config.face_model)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    path,
                )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                output_face_blendshapes=True,
                num_faces=10,
            )
            self.cache["face"] = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        result = self.cache["face"].detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        )
        faces = []
        for landmarks, blendshapes in zip(result.face_landmarks, result.face_blendshapes):
            xs, ys = [p.x for p in landmarks], [p.y for p in landmarks]
            scores = {b.category_name: float(b.score) for b in blendshapes}
            width = (max(xs) - min(xs)) * image.shape[1]
            expression = "unknown"
            if (
                width >= 48
                and min(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0)) > 0.5
            ):
                expression = "smiling_looking"
            faces.append(
                {
                    "bbox": [
                        min(xs) * image.shape[1],
                        min(ys) * image.shape[0],
                        max(xs) * image.shape[1],
                        max(ys) * image.shape[0],
                    ],
                    "blendshapes": scores,
                    "expression": expression,
                }
            )
        return faces
