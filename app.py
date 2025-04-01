import os
import cv2
import pytesseract
import numpy as np
from flask import Flask, request, send_file
from PIL import ImageFont, ImageDraw, Image
from insightface.app import FaceAnalysis

# Initialize Flask and face analysis model
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function for face swapping with blending
def swap_face_in_id(id_img, new_face_img, x1, y1, x2, y2):
    new_face_resized = cv2.resize(new_face_img, (x2 - x1, y2 - y1))

    # Create a mask for seamless cloning
    mask = 255 * np.ones(new_face_resized.shape, new_face_resized.dtype)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    id_img = cv2.seamlessClone(new_face_resized, id_img, mask, center, cv2.NORMAL_CLONE)

    return id_img

# Route to handle face swapping
@app.route("/swap", methods=["POST"])
def swap_face():
    id_image = request.files["id_image"]
    new_face = request.files["new_face"]

    # Save uploaded images temporarily
    id_path = os.path.join(UPLOAD_FOLDER, id_image.filename)
    face_path = os.path.join(UPLOAD_FOLDER, new_face.filename)
    id_image.save(id_path)
    new_face.save(face_path)

    # Load images
    id_img = cv2.imread(id_path)
    new_face_img = cv2.imread(face_path)

    # Detect faces in the ID
    id_faces = face_app.get(id_img)

    if len(id_faces) > 0:
        x1, y1, x2, y2 = id_faces[0].bbox.astype(int)

        # Perform face swap with blending
        swapped_img = swap_face_in_id(id_img, new_face_img, x1, y1, x2, y2)

        # Save and send the final image
        swapped_path = os.path.join(UPLOAD_FOLDER, "swapped_id.jpg")
        cv2.imwrite(swapped_path, swapped_img)

        return send_file(swapped_path, mimetype="image/jpeg")

    return "No face detected in ID!", 400

# Route to detect text positions on the ID
@app.route("/detect_text_positions", methods=["POST"])
def detect_text_positions():
    id_path = os.path.join(UPLOAD_FOLDER, "swapped_id.jpg")
    
    # Load the image
    img = cv2.imread(id_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract text positions using OCR
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    text_positions = []
    for i in range(len(text_data["text"])):
        if text_data["text"][i].strip():
            x, y, w, h = (text_data["left"][i], text_data["top"][i], text_data["width"][i], text_data["height"][i])
            text_positions.append({"text": text_data["text"][i], "x": x, "y": y, "w": w, "h": h})

    return {"positions": text_positions}

# Route to edit text dynamically
@app.route("/edit_text_dynamic", methods=["POST"])
def edit_text_dynamic():
    new_text = request.form["new_text"]
    original_text = request.form["original_text"]
    id_path = os.path.join(UPLOAD_FOLDER, "swapped_id.jpg")

    # Load the image
    img = cv2.imread(id_path)

    # Detect text positions
    positions = detect_text_positions()["positions"]
    
    # Find the position of the original text
    for pos in positions:
        if pos["text"].lower() == original_text.lower():
            x, y, w, h = pos["x"], pos["y"], pos["w"], pos["h"]
            break
    else:
        return "Original text not found!", 400

    # Set up PIL for better text rendering
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Choose a close font
    font_path = "arial.ttf"  # Adjust if needed
    font = ImageFont.truetype(font_path, h)

    # Overlay new text in the same position
    draw.text((x, y), new_text, font=font, fill=(0, 0, 0))

    # Convert back to OpenCV format
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    final_path = os.path.join(UPLOAD_FOLDER, "final_id.jpg")
    cv2.imwrite(final_path, final_img)

    return send_file(final_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)

