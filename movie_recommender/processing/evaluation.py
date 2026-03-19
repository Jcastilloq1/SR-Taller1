from processing.recommender import evaluate_model

print("User-User Cosine RMSE:", evaluate_model("cosine", True))
print("User-User Pearson RMsSE:", evaluate_model("pearson", True))
print("User-User MSD RMSE:", evaluate_model("msd", True))

print("Item-Item Cosine RMSE:", evaluate_model("cosine", False))
print("Item-Item Pearson RMSE:", evaluate_model("pearson", False))
print("Item-Item MSD RMSE:", evaluate_model("msd", False))