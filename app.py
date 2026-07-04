from flask import Flask, request, jsonify
from rag import ask_loan_assistant

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Loan Officer AI API is running."
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is missing."
            }), 400

        question = data.get("question")

        if not question:
            return jsonify({
                "error": "Question is required."
            }), 400

        answer = ask_loan_assistant(question)

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=3000,
        debug=True
    )