from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Input form

@app.route("/predict", methods=["POST"])
def predict_data():
    if request.method == "POST":
        
        # Collect user input from form
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("race_ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            reading_score = float(request.form.get("reading_score")),
            writing_score = float(request.form.get("writing_score"))
        )

        final_data = data.get_data_as_dataframe()

        # Load model & make prediction
        predict = PredictPipeline()
        result = predict.predict(final_data)

        return render_template("home.html", prediction=result[0])

if __name__ == "__main__":
    app.run(debug=True)
