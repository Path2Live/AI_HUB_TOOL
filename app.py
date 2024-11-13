from flask import Flask, request, jsonify
from helperclass import DataLoader, chat_completion, extract_project_info, extract_cost_info, formulate_question
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/report', methods=['POST'])
def report_details():
    try:
        uploaded_files = request.files.getlist("files")
        pdf_paths = []

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                pdf_paths.append(file_path)

        if not pdf_paths:
            return jsonify({"error": "No valid PDF files uploaded"}), 400

        json_data = request.form.get('data')
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        data = json.loads(json_data)
        project_info_payload = data.get('project_info_payload', [])
        cost_info_payload = data.get('cost_info_payload', [])

        documents_content = DataLoader.load_pdf_contents(pdf_paths)
        api_key = os.getenv("OPENAI_API_KEY")

        project_details = extract_project_info(project_info_payload)
        cost_info = extract_cost_info(cost_info_payload)
        question = formulate_question(project_details, cost_info, historical_data=DataLoader.load_historical_data())

        messages = [
            {"role": "system", "content": "You are a QC and architect."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": documents_content}
        ]

        response = chat_completion(messages, api_key)

        for file_path in pdf_paths:
            os.remove(file_path)

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
