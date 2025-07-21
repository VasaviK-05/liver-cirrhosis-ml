from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("best_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        alcohol = float(request.form['alcohol'])
        hep_b = float(request.form['hep_b'])
        hep_c = float(request.form['hep_c'])
        diabetes = float(request.form['diabetes'])
        obesity = float(request.form['obesity'])
        ldl = float(request.form['ldl'])
        hdl = float(request.form['hdl'])
        tg = float(request.form['tg'])
        tch = float(request.form['tch'])

        input_data = np.array([[age, gender, alcohol, hep_b, hep_c,
                                diabetes, obesity, ldl, hdl, tg, tch]])

        prediction = model.predict(input_data)

        result = "Liver Cirrhosis Detected" if prediction[0] == 1 else "No Cirrhosis Detected"
        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
