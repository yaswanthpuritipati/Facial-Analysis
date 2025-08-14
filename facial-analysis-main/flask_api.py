from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    image = request.files.get("image")
    
    if not name or not image:
        return jsonify({"error": "Name and image are required"}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, f"{name}.jpeg")
    image.save(image_path)
    
    try:
        subprocess.run(["python3", "main4.py", "--register", name, "--register-image", image_path], check=True)
        return jsonify({"message": f"User {name} registered successfully"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to register user: {str(e)}"})

@app.route("/predict", methods=["GET"])
def predict():
    try:
        subprocess.run(["python3", "main4.py", "--source", "0"], check=True)
        return jsonify({"message": "Prediction started"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to start prediction: {str(e)}"})

@app.route("/predict_image", methods=["POST"])
def predict_image():
    print(f"Received request method: {request.method}")
    print(f"Headers: {request.headers}")
    
    if request.method != "POST":
        return jsonify({"error": "Invalid request method"}), 405
    
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, "prediction_image.jpeg")
    image.save(image_path)
    
    try:
        print(f"Running prediction command: python3 main4.py --source {image_path}")
        result = subprocess.run(["python3", "main4.py", "--source", image_path], check=True, capture_output=True, text=True)
        output = result.stdout.strip()
        print(f"Raw Prediction Output: {output}")
        
        # Parse expected output format (gender, age, emotion, name)
        parsed_output = {}
        lines = output.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_output[key.strip()] = value.strip()
        
        if not parsed_output:
            return jsonify({"error": "No valid prediction data returned from main4.py"})
        
        return jsonify({"message": "Image prediction completed", "prediction": parsed_output})
    except subprocess.CalledProcessError as e:
        print(f"Prediction failed: {e.stderr}")
        return jsonify({"error": f"Prediction failed: {str(e.stderr)}"})

if __name__ == "__main__":
    print("Flask API is running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)