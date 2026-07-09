import os
import io
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pypdf import PdfReader
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from rag_engine import rag_instance

load_dotenv()

app = Flask(__name__)

# Standard structural dictionary format for the updated watsonx.ai SDK
WATSONX_CREDENTIALS = {
    "url": "https://au-syd.ml.cloud.ibm.com",
    "apikey": os.environ.get("IBM_API_KEY", "").strip()
}
PROJECT_ID = os.environ.get("IBM_PROJECT_ID", "").strip()

model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=WATSONX_CREDENTIALS,
    project_id=PROJECT_ID,
    params={
        "decoding_method": "sample",
        "max_new_tokens": 800,
        "temperature": 0.3,
        "top_p": 0.9,
    },
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    # 1. Fetch text variables from Multi-part form layout
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "message field is required"}), 400

    # 2. Intercept and unpack binary PDF streams directly from memory
    user_resume_text = ""
    if "resume" in request.files:
        uploaded_file = request.files["resume"]
        if uploaded_file.filename != "" and uploaded_file.filename.lower().endswith(".pdf"):
            try:
                # Wrap binary content in a non-blocking in-memory stream object
                pdf_stream = io.BytesIO(uploaded_file.read())
                reader = PdfReader(pdf_stream)
                
                extracted_pages = []
                for page in reader.pages:
                    text_content = page.extract_text()
                    if text_content:
                        extracted_pages.append(text_content)
                
                user_resume_text = "\n".join(extracted_pages)
                print(f"Successfully processed uploaded resume context: ({len(user_resume_text)} characters extracted)")
            except Exception as e:
                print(f"Structural resume text ingestion failure: {str(e)}")

    # 3. Perform localized multi-dimensional matrix similarity dot-product checks
    retrieved_knowledge = rag_instance.retrieve_relevant_context(user_message, top_k=3)

    # 4. Synthesize context layout inside strict domain instructions
    system_instruction = (
        "You are an elite AI-Powered Interview Trainer Agent operating within a native watsonx.ai RAG pipeline.\n"
        "Analyze the candidate's metrics and resume data to deliver precise feedback.\n\n"
        "--- GROUNDING KNOWLEDGE BASE DIRECTIVES ---\n"
        f"{retrieved_knowledge}\n\n"
        "--- CANDIDATE RESUME PROFILE (AUTOMATICALLY EXTRACTED) ---\n"
        f"{user_resume_text if user_resume_text else '[No user resume document uploaded yet]'}\n\n"
        "--- STRICT DOMAIN GUARDRAILS ---\n"
        "1. You are strictly an interview preparation, career coaching, and technical/behavioral mock simulation agent.\n"
        "2. If the user asks an out-of-domain question (e.g., general science, history, space, pop culture, cooking, etc.) "
        "that is completely unrelated to interviews, careers, or engineering skills, you must politely decline to answer. "
        "State that your system is strictly constrained to interview training and prompt them to return to career preparation."
    )

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instruction}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    result = model.generate_text(prompt=prompt)
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)