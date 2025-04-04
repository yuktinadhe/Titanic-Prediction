# 🚢 Titanic Survival Prediction | Machine Learning Project

Predicting survival on the Titanic using powerful classification algorithms.

This project uses the famous **Titanic dataset** to build a machine learning model that predicts whether a passenger survived or not based on features like age, sex, ticket class, and more.

---

## 🎯 Business Problem

The goal is to create a predictive model that determines which passengers are more likely to survive the Titanic disaster.

By analyzing passenger details such as:
- **Age**
- **Sex**
- **Ticket Class**
- **Siblings/Spouse aboard**
- **Parents/Children aboard**
- **Fare**
- **Embarked Port**

We can predict the **`Survived`** status (0 = No, 1 = Yes) using classification algorithms.

---

## 📊 Dataset Features

| Feature      | Description                                  |
|--------------|----------------------------------------------|
| `PassengerId`| Unique ID of each passenger                  |
| `Survived`   | Survival (0 = No, 1 = Yes)                   |
| `Pclass`     | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)     |
| `Name`       | Name of the passenger                        |
| `Sex`        | Gender                                       |
| `Age`        | Age in years                                 |
| `SibSp`      | # of siblings/spouses aboard                 |
| `Parch`      | # of parents/children aboard                 |
| `Ticket`     | Ticket number                                |
| `Fare`       | Passenger fare                               |
| `Cabin`      | Cabin number                                 |
| `Embarked`   | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🧠 Models Used

Implemented and evaluated the following models:

- ✅ Logistic Regression  
- ✅ K-Nearest Neighbors (KNN)  
- ✅ Support Vector Machine (SVM)  
- ✅ Decision Tree  
- ✅ Random Forest  
- ✅ XGBoost ⭐️ (Best Model - 90% Accuracy)  
- ✅ AdaBoost  
- ✅ Gradient Boosting  
- ✅ Artificial Neural Network (ANN)

---

## 📈 Best Model: XGBoost

Achieved **~90% accuracy** using **XGBoost**, which performed the best after feature engineering and hyperparameter tuning.

---

## 📌 Tech Stack

| Tool/Library    | Purpose                      |
|------------------|------------------------------|
| Python 🐍        | Main language                |
| Pandas & NumPy 📊| Data manipulation            |
| Matplotlib & Seaborn 📈 | Visualizations     |
| Scikit-learn ⚙️ | ML models + Evaluation       |
| XGBoost 💥       | Gradient boosting classifier |
| TensorFlow/Keras 🤖 | ANN Model (optional)     |


---

📊 Evaluation Metrics
- Accuracy

- Precision

- Recall

- F1 Score

- Confusion Matrix

- ROC AUC Curve

---

Future Work
- ✅ Deploy as a web app using Streamlit or Flask

- ✅ Use GridSearchCV for hyperparameter tuning

- ✅ Create a REST API for real-time prediction

- ✅ Experiment with Deep Learning further

---

🙋‍♀️ Author
👩‍💻 Developed by Yukti Nadhe
