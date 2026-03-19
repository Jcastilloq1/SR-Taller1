from flask import Flask, render_template, request
from processing.engine import get_recommendations, get_user_history

app = Flask(__name__, template_folder="web/templates")


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    user_id = int(request.form["user_id"])
    method = request.form["method"]
    similarity = request.form["similarity"]

    history = get_user_history(user_id)

    recs = get_recommendations(user_id, method, similarity)

    return render_template(
        "results.html",
        user=user_id,
        method=method,
        similarity=similarity,
        history=history.to_dict(orient="records"),
        recs=recs.to_dict(orient="records")
    )


if __name__ == "__main__":

    app.run(debug=True)