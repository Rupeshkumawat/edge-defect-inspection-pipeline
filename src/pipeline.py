import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

def letterbox(
    img: np.ndarray, 
    new_shape: Tuple[int, int] = (640, 640), 
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)

def preprocess(img_bytes: bytes) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
    nparr = np.frombuffer(img_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise ValueError("Invalid or corrupted image format")

    orig_shape = orig_img.shape[:2]
    img, ratio, (dw, dh) = letterbox(orig_img, (640, 640))
    
    # BGR to RGB, scale to [0.0, 1.0], transpose HWC -> CHW
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(img, (2, 0, 1))
    return tensor, ratio, (dw, dh), orig_shape

def postprocess(
    predictions: np.ndarray, 
    ratio: float, 
    pad: Tuple[float, float], 
    orig_shape: Tuple[int, int],
    conf_threshold: float = 0.45, 
    iou_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    preds = np.squeeze(predictions)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4:]
    
    max_scores = np.max(scores, axis=1)
    class_assignments = np.argmax(scores, axis=1)

    mask = max_scores >= conf_threshold
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    max_scores = max_scores[mask]
    class_assignments = class_assignments[mask]

    x1 = cx - (w / 2)
    y1 = cy - (h / 2)

    boxes_for_nms = [[int(x), int(y), int(width), int(height)] for x, y, width, height in zip(x1, y1, w, h)]
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, max_scores.tolist(), conf_threshold, iou_threshold)

    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw, bh = boxes_for_nms[idx]
            orig_x1 = max(0, int((bx - pad[0]) / ratio))
            orig_y1 = max(0, int((by - pad[1]) / ratio))
            orig_x2 = min(orig_shape[1], int((bx + bw - pad[0]) / ratio))
            orig_y2 = min(orig_shape[0], int((by + bh - pad[1]) / ratio))

            results.append({
                "class_id": int(class_assignments[idx]),
                "confidence": round(float(max_scores[idx]), 3),
                "bbox": [orig_y1, orig_x1, orig_y2, orig_x2]
            })

    return results
