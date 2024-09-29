import numpy as np
import cv2
from skin_analysis.darkcircle_measurement import extract_dark_circle_skin_tone, combine_masks_and_calculate_area

# Define color_map for 6 classes
color_map = {
    0: [255, 50, 50],       # BGR color
    1: [106, 106, 106],     # BGR color
    2: [255, 196, 61],      # BGR color
    3: [203, 125, 247],      # BGR color
    4: [203, 125, 247],      # BGR color
    5: [250, 140, 195],     # BGR color
}

def is_inside(mask1, mask2):
    """Check if mask1 is completely inside mask2."""
    return np.all(mask2[mask1 > 0])

def infer_and_visualize_per_class(image, image2, face, stone, predictor):
    im = image.copy()  # Make a copy of the original image for overlay
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()

    result = {}

    # Handle dark_circle detection
    dark_circle_combined_mask, dark_circle_total_area = combine_masks_and_calculate_area(instances, pred_classes, 1)
    if dark_circle_total_area > 0:
        dark_circle = extract_dark_circle_skin_tone(image, stone, predictor)
        result["darkcircle"] = dark_circle  

        # Draw the combined dark_circle mask on the image
        contours, _ = cv2.findContours(dark_circle_combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.polylines(image2, [contour], isClosed=True, color=color_map.get(1, [255, 255, 255]), thickness=2)

    # Handle wrinkle detection (class 3 and 4)
    wrinkle_combined_mask_3, wrinkle_total_area_3 = combine_masks_and_calculate_area(instances, pred_classes, 3)
    wrinkle_combined_mask_4, wrinkle_total_area_4 = combine_masks_and_calculate_area(instances, pred_classes, 4)
    
    total_wrinkle_area = wrinkle_total_area_3 + wrinkle_total_area_4
    if total_wrinkle_area > 0:

        # Draw the combined wrinkle masks (both classes 3 and 4)
        combined_wrinkle_mask = np.maximum(wrinkle_combined_mask_3, wrinkle_combined_mask_4)
        contours, _ = cv2.findContours(combined_wrinkle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.polylines(image2, [contour], isClosed=True, color=color_map.get(3, [255, 255, 255]), thickness=2)
        
        wrinkle_count = len(contours)
        wrinkle_score = (10 - ((total_wrinkle_area / face) * 10)) - (wrinkle_count * 0.1)
        # Ensure wrinkle_score is not negative
        wrinkle_score = max(wrinkle_score, 0)
        result["wrinkle"] = wrinkle_score

    # Process other classes
    class_name_map = {
        0: "acne",
        2: "pore",
        5: "pigmentation_melanin"
    }

    for cls, class_name in class_name_map.items():
        class_combined_mask, total_class_area = combine_masks_and_calculate_area(instances, pred_classes, cls)
        if total_class_area > 0:
            class_score_area = 10 - ((total_class_area / face) * 10)
            result[class_name] = class_score_area

            # Draw the masks for the class on the image
            contours, _ = cv2.findContours(class_combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.polylines(image2, [contour], isClosed=True, color=color_map.get(cls, [255, 255, 255]), thickness=2)

    return result, image2
