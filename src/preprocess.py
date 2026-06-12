import cv2
import numpy as np

def apply_clahe(img):
    # 0~255 스케일로 변환
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(img_uint8)
    return result.astype(np.float32) / 255.0

def apply_binary(img):
    # 0~255 스케일로 변환
    img_uint8 = (img * 255).astype(np.uint8)
    _, result = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
    return result.astype(np.float32) / 255.0

def apply_canny(img):
    # 0~255 스케일로 변환
    img_uint8 = (img * 255).astype(np.uint8)
    result = cv2.Canny(img_uint8, 50, 150)
    return result.astype(np.float32) / 255.0

def preprocess(img, method='baseline'):
    if method == 'baseline':
        return img
    elif method == 'clahe':
        return apply_clahe(img)
    elif method == 'binary':
        return apply_binary(img)
    elif method == 'edge':
        return apply_canny(img)
    else:
        raise ValueError(f"Unknown method: {method}")
