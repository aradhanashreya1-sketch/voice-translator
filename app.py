from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

KAGGLE_URL_FILE = "/var/www/voice-translator/kaggle_url.txt"

def get_kaggle_url():
    try:
        with open(KAGGLE_URL_FILE, "r") as f:
            return f.read().strip()
    except:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/kaggle-url", methods=["POST"])
def update_kaggle_url():
    data = request.json
    url = data.get("url")
    if url:
        with open(KAGGLE_URL_FILE, "w") as f:
            f.write(url)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route("/api/status")
def status():
    url = get_kaggle_url()
    return jsonify({
        "kaggle_url": url,
        "active": url is not None
    })

if __name__ == "__main__":
    app.run(debug=True)
