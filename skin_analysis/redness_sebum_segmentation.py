import torch
import cv2
import numpy as np
from utils.face_segmentation import CaptureFrames
from ultralytics import YOLO
from model.models import LinkNet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LinkNet34()
model.load_state_dict(torch.load('model/linknet.pth', map_location=lambda storage, loc: storage))
model.eval()
model.to(device)

def process_mask_image(masked_face_image, face_mask, orig_image):
    model = YOLO('model/redness.pt')
    masked_face_area = cv2.bitwise_and(masked_face_image, masked_face_image, mask=face_mask.astype(np.uint8))
    results = model.predict(source=masked_face_area, conf=0.5, iou=0)
    
    total_redness_area = 0
    total_face_area = np.sum(face_mask > 0)  # Total face area in pixels
    orig_height, orig_width = masked_face_image.shape[:2]
    overlay = orig_image.copy()

    if results[0].masks is not None:
        for i, r in enumerate(results[0].masks.data):
            obj_mask = r.cpu().numpy()
            obj_mask_resized = cv2.resize(obj_mask, (orig_width, orig_height))
            redness_area = np.sum(obj_mask_resized > 0.5)  # Area of redness in pixels
            total_redness_area += redness_area
            overlay[obj_mask_resized > 0.5] = (70, 70, 255)  # Redness highlighted in red

        final_output = cv2.addWeighted(overlay, 0.5, orig_image, 0.5, 0)
    else:
        final_output = orig_image.copy()

    return final_output, total_redness_area, total_face_area  # Return the output image and area

def shine3(orig_image, masked_face_image, face_mask):
    masked_face_area = cv2.bitwise_and(masked_face_image, masked_face_image, mask=face_mask.astype(np.uint8))
    gray = cv2.cvtColor(masked_face_area, cv2.COLOR_BGR2GRAY)
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm_gray, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    overlay = orig_image.copy()
    overlay[morph == 255] = [0, 165, 255]  # Highlight shine in orange

    final_output = cv2.addWeighted(overlay, 0.8, orig_image, 0.2, 0)

    total_shine_area = np.sum(morph == 255)  # Total shine area in pixels
    total_face_area = np.sum(face_mask > 0)  # Total face area in pixels

    return final_output, total_shine_area, total_face_area  # Return output image, shine area, and face area

def main(image):
    orig_image = image.copy()

    # Process the image to get the masked face and face mask
    capture = CaptureFrames(model, show_mask=True)
    masked_face_image, face_mask = capture(orig_image)

    if masked_face_image is None or face_mask is None:
        return None, None, None

    # Redness detection
    result_image_redness, total_redness_area, total_face_area = process_mask_image(masked_face_image, face_mask, orig_image)

    # Shine detection
    result_image_sebum, total_shine_area, _ = shine3(orig_image, masked_face_image, face_mask)

    total_redness_area = 10 - ((total_redness_area / total_face_area) * 10)
    total_shine_area = 10 - ((total_shine_area / total_face_area) * 10)

    return result_image_redness, result_image_sebum, total_redness_area, total_shine_area, total_face_area
