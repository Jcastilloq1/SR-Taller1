from flask import Flask,render_template,request
import sys

sys.path.append("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/processing")

from recommender_system import recommend
from model_evaluator import get_all_rmse

app=Flask(__name__)


@app.route("/")
def home():

    rmses=get_all_rmse()

    return render_template("index.html",rmses=rmses)


@app.route("/recommend",methods=["POST"])
def rec():

    user=int(request.form["user"])
    method=request.form["method"]
    similarity=request.form["similarity"]

    results=recommend(user,method,similarity)

    return render_template("results.html",
                           results=results,
                           user=user,
                           method=method,
                           similarity=similarity)


app.run(debug=True)