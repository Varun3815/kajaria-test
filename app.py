from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH
from utils import allowed_file
from image_matcher import find_best_matches
from clip_matcher import search_tiles_by_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Health check endpoint for ALB
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# ✅ Actual API endpoint for frontend
@app.route('/api/search', methods=['POST', 'GET'])
def search_tiles():
    if request.method == 'GET':
        return jsonify({"status": "healthy"}), 200  # fallback for GET health check if needed

    matches = []
    filename = None

    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            matches = find_best_matches(filepath)

            # Delete uploaded files after processing
            for f in os.listdir(UPLOAD_FOLDER):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                except PermissionError:
                    pass

    elif 'description' in request.form:
        description = request.form['description'].strip()
        if description:
            matches = search_tiles_by_text(description)

    # Fix float32 serialization
    matches = [(name, float(score)) for name, score in matches]

    return jsonify({
        "filename": filename,
        "matches": matches
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
