import os
import shutil
import time

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import ingest_document, get_qa_chain, release_vectordb

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"pdf", "csv", "txt", "doc", "docx", "json", "md"}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        num_chunks = ingest_document(path)
        return jsonify({
            "message": f"Processed into {num_chunks} chunks.",
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        chain = get_qa_chain()
        answer = chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    filename = data.get("filename", "").strip()

    try:
        # Release ChromaDB lock first
        release_vectordb()
        time.sleep(0.5)

        # Delete the uploaded file
        if filename:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete the vector database
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

        return jsonify({"message": "Document deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)