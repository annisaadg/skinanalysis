import numpy as np
import cv2

def combine_masks_and_calculate_area(instances, pred_classes, target_class):
    """Combine all masks for the given class and calculate the total area."""
    class_instances = instances[pred_classes == target_class]
    
    if len(class_instances) == 0 or len(class_instances.pred_masks) == 0:
        return np.zeros((instances.image_size[0], instances.image_size[1]), dtype=np.uint8), 0

    combined_mask = np.zeros_like(class_instances.pred_masks[0].numpy(), dtype=np.uint8)
    total_area = 0
    for mask in class_instances.pred_masks:
        combined_mask = np.maximum(combined_mask, mask.numpy().astype(np.uint8))
        total_area += np.sum(mask.numpy())
    return combined_mask, total_area

def extract_dark_circle_skin_tone(image, stone, predictor):
    im = image.copy()
    outputs = predictor(im)

    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()

    # Combine dark_circle masks
    dark_circle_combined_mask, total_area = combine_masks_and_calculate_area(instances, pred_classes, 1)  # Class 1: Dark Circle

    if total_area == 0:
        print("No Dark Circle detected.")
        return None

    # Extract dark_circle region from the image
    extracted_region = cv2.bitwise_and(im, im, mask=dark_circle_combined_mask)
    extracted_region_gray = cv2.cvtColor(extracted_region, cv2.COLOR_BGR2GRAY)
    mean_color = int(np.mean(extracted_region_gray[dark_circle_combined_mask > 0]))

    # Convert the mean color to HEX
    hex_color = "#{:02x}{:02x}{:02x}".format(mean_color, mean_color, mean_color)

    # Measure darkness based on the skin tone and dark circle color
    score = measure_darkness(hex_color, stone)
    return score

def hex_to_rgb(hex_color):
    """Convert HEX color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def calculate_luminance(rgb_color):
    """Calculate luminance from RGB color."""
    r, g, b = rgb_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance

def measure_darkness(darkcircle_hex, skintone_hex):
    """Measure the difference in luminance between skin tone and dark circle."""
    darkcircle_rgb = hex_to_rgb(darkcircle_hex)
    skintone_rgb = hex_to_rgb(skintone_hex)

    darkcircle_luminance = calculate_luminance(darkcircle_rgb)
    skintone_luminance = calculate_luminance(skintone_rgb)

    darkness_difference = abs(skintone_luminance - darkcircle_luminance)

    score = 10 - ((darkness_difference / skintone_luminance) * 10)
    
    return score
