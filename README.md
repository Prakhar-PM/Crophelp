Crop Yield Prediction and Recommendation System

Overview

The Crop Yield Prediction and Recommendation System is a web application that utilizes machine learning models to predict crop yields and recommend suitable crops for cultivation based on user-provided inputs. It is implemented using Flask for the web backend and Scikit-learn for machine learning.

Features

Crop Yield Prediction:

Predict the yield of a crop based on factors such as year, rainfall, pesticide usage, average temperature, area, and crop type.

Crop Recommendation:

Recommend the best crop to cultivate based on environmental factors like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

User-Friendly Interface:

An intuitive web form built using Bootstrap for data input and displaying predictions.

Model Evaluation:

Compares the performance of multiple machine learning models, selecting the most accurate one for predictions.

Technologies Used

Backend:

Python

Flask

Frontend:

HTML

CSS (Bootstrap Framework)

Machine Learning:

Scikit-learn

Models:

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

Decision Tree

Random Forest

Bagging, AdaBoost, and Gradient Boosting Classifiers

Installation and Setup

Prerequisites:

Python 3.8+

Flask

Scikit-learn

Pandas

Numpy

Steps:

Clone the Repository:

git clone https://github.com/your-repository/crop-yield-prediction.git
cd crop-yield-prediction

Install Dependencies:

pip install -r requirements.txt

Prepare the Dataset:

Ensure the Crop_recommendation.csv file is available in the project directory.

Train the Model:
Run the training script to generate the model and scaler files:

python train_model.py

This will save the trained model (model.pkl) and scaler (scaler.pkl) files.

Start the Flask Application:

python app.py

Access the Application:
Open your browser and navigate to http://127.0.0.1:5000/.

Usage

Crop Yield Prediction:

Enter values for year, rainfall, pesticides, temperature, area, and crop type.

Click on "Predict" to get the predicted yield.

Crop Recommendation:

Modify the recommendation() function inputs (N, P, K, temperature, etc.) and run the script.

The recommended crop will be printed to the console.

Project Structure

.
├── app.py               # Flask application
├── templates/
│   └── index.html       # HTML template
├── static/
│   └── styles.css       # Optional custom styles
├── Crop_recommendation.csv  # Dataset
├── model.pkl            # Trained model
├── scaler.pkl           # Preprocessing scaler
├── train_model.py       # Script for training and saving the model
└── requirements.txt     # Python dependencies

Model Details

Random Forest Classifier:

Final model used for crop recommendation.

Selected for its high accuracy and performance in testing.

Preprocessing:

Min-Max Scaling is applied to normalize the features.

Future Enhancements

Integrate Real-Time Data:

Fetch weather and soil data dynamically using APIs.

Advanced Models:

Experiment with deep learning models for better accuracy.

Improved UI/UX:

Enhance the frontend for better interactivity and responsiveness.

Error Handling:

Add more robust error handling and logging mechanisms.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

License

This project is licensed under the MIT License. See the LICENSE file for details.# Crophelp
