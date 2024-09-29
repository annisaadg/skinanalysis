from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from model.skinanalysis_model import SkinModel
from skin_analysis.skintone_recognition import StoneModel
from skin_analysis.age_recognition import age_recognition
from skin_analysis.redness_sebum_segmentation import main as redness_shine_main
from skin_analysis.visualization import infer_and_visualize_per_class
import numpy as np

app = Flask(__name__)
skin_model = SkinModel()
stone_model = StoneModel()

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Load image
        image = cv2.imread(file_path)

        # Run skin tone detection
        skin_tone_result, skin_tone_code = stone_model.detect_skin_tone(file_path)

        # Run redness and shine detection
        result_image_redness, result_image_sebum, red, shine, face = redness_shine_main(image)

        # Combine images
        combined_result_image = cv2.addWeighted(result_image_redness, 0.5, result_image_sebum, 0.5, 0)

        # Save the combined images
        result_image_path = os.path.join(OUTPUT_FOLDER, 'redness_sebum.jpg')
        cv2.imwrite(result_image_path, combined_result_image)

        image2 = cv2.imread(result_image_path)

        # Run skin analysis and get the image with segmentation
        analysis_result, skin_analysis = infer_and_visualize_per_class(image, image2, face, skin_tone_code, skin_model.predictor)

        acne_score = analysis_result.get('acne', 10)
        wrinkle_score = analysis_result.get('wrinkle', 10)
        dark_circle_score = analysis_result.get('darkcircle', 10)
        pigmentation_score = analysis_result.get('pigmentation_melanin', 10)
        open_pores_score = analysis_result.get('pore', 10)

        score = ((acne_score+wrinkle_score+dark_circle_score+pigmentation_score+open_pores_score+red+shine)/7)

        age = age_recognition(image, score, wrinkle_score, pigmentation_score)

        # Save the skin_analysis image
        skin_analysis_path = os.path.join(OUTPUT_FOLDER, 'skin_analysis.jpg')
        cv2.imwrite(skin_analysis_path, skin_analysis)

        # Combine all results into one final output
        result = {
            "skin_age": age,
            "acne" : acne_score,
            "darkcircle" : dark_circle_score,
            "wrinkle" : wrinkle_score,
            "pore" : open_pores_score,
            "sebum": shine,
            "skin_core" : score,
            "pigmentation_melanin" : pigmentation_score,
            "skin_tone": skin_tone_result,
            "hemoglobin": red,
            "skin_analysis_image_result": skin_analysis_path
        }

        # Convert the result dictionary to a serializable format
        serializable_result = convert_to_serializable(result)

        return jsonify(serializable_result)

if __name__ == '__main__':
    app.run(debug=True)