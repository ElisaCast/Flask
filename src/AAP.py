from flask import Flask
from flask import Flask, render_template, request
import numpy as np
import joblib
import gunicorn

"""#Importamos el objeto Flask
app = Flask(__name__) # Variable especial name que contiene el nombre del módulo de Python actual

#Función de respuesta HTTP
@app.route('/') # Decorador que convierte una función normal de Python en una función de vista Flask
def hello():
    return 'Hello, World!'"""

# Puedo acceder a tu aplicación Flask abriendo un navegador web y navegando a http://127.0.0.1:5000 o http://localhost:5000.
# He utilizado Waitres porque  Gunicorn depende de ciertas características específicas de Unix, y ejecutarlo directamente en
# un entorno Windows puede ocasionar problemas.

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('C:/Users/elisa/OneDrive/Escritorio/Data Sciencie/ML-BosstingAlgoritmo/models/boosting_classifier_nestimators-20_learnrate-0.001_42.sav')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Obtener los valores del formulario
        val1 = float(request.form['Insulin'])
        val2 = float(request.form['Age'])
        val3 = float(request.form['Pregnancies'])
        val4 = float(request.form['DiabetesPedigreeFunction'])
        val5 = float(request.form['BMI'])
        val6 = float(request.form['BloodPressure'])
        val7 = float(request.form['Glucose'])
        val8 = float(request.form['SkinThickness'])

        # Realizar la predicción utilizando el modelo
        input_data = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)