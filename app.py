from flask import Flask, render_template, redirect, url_for
import subprocess
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route("/")
def index():
    runs_data = []
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("DVC_MLflow_Integrated_Project")
        if experiment:
            runs = client.search_runs(experiment.experiment_id)
            for r in runs:
                runs_data.append({
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "accuracy": r.data.metrics.get("accuracy", "N/A"),
                    "dataset_version": r.data.params.get("dataset_version", "N/A"),
                    "n_estimators": r.data.params.get("n_estimators", "N/A")
                })
    except Exception as e:
        print("Error fetching mlflow runs:", e)

    return render_template("index.html", runs=runs_data)

@app.route("/train", methods=["POST"])
def train():
    subprocess.run(["python", "train_integrated.py"], check=True)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
