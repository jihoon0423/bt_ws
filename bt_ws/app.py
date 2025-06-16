# backend/app.py
from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)
IMAGE_DIR = os.path.expanduser("~/waypoint_photos")

@app.route("/api/images")
def get_image_sets():
    photo_sets = []
    for fname in os.listdir(IMAGE_DIR):
        if fname.startswith("waypoint_") and fname.endswith(".png"):
            idx = fname.split('_')[1].split('.')[0]
            current = f"waypoint_{idx}.png"
            reference = f"reference{idx}.png"
            diff = f"comparison_result_{idx}.png"
            if all(os.path.exists(os.path.join(IMAGE_DIR, f)) for f in [reference, diff]):
                photo_sets.append({
                    "idx": idx,
                    "current": f"/images/{current}",
                    "reference": f"/images/{reference}",
                    "diff": f"/images/{diff}"
                })
    return jsonify(photo_sets)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
