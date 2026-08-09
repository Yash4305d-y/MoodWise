# 🧠 MoodWise

> **My First NLP Project — Understand the emotion behind the words.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154F5C?style=for-the-badge)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

### 🌐 [Try MoodWise Live](https://moodwise.streamlit.app/)

---

## 🌱 My First NLP Project

**MoodWise is my first hands-on Natural Language Processing (NLP) project.**

I built this project while learning the fundamentals of Machine Learning and NLP, with the goal of understanding how machines can process human language and identify the emotion expressed in text.

Rather than starting with complex deep-learning or transformer architectures, MoodWise focuses on understanding the **fundamentals of traditional NLP and machine learning**:

```text
Raw Text
    ↓
Text Preprocessing
    ↓
Feature Extraction
    ↓
Machine Learning
    ↓
Emotion Prediction
```

Building MoodWise gave me practical experience taking an NLP idea from **data preprocessing and experimentation in Jupyter Notebook to a deployed Streamlit application**.

---

## ✨ Overview

MoodWise is an NLP-based machine learning application that analyzes text and predicts the emotion expressed in it.

The project experiments with two text representation techniques:

* **Bag of Words**
* **TF-IDF**

and compares them using:

* **Multinomial Naive Bayes**
* **Logistic Regression**

The best-performing combination was **Logistic Regression + Bag of Words**, achieving **88.84% accuracy** on the test set.

---

## 📊 Model Performance

| Model                   | Vectorization |      Accuracy |
| ----------------------- | ------------- | ------------: |
| Multinomial Naive Bayes | Bag of Words  |        76.78% |
| Multinomial Naive Bayes | TF-IDF        |        66.09% |
| Logistic Regression     | Bag of Words  | **88.84% 🏆** |
| Logistic Regression     | TF-IDF        |        86.06% |

### 🏆 Best Model

**Logistic Regression + Bag of Words**

**Accuracy: 88.84%**

---

## 🌐 Live Demo

### 🚀 Try MoodWise

**https://moodwise.streamlit.app/**

The trained NLP model is deployed using Streamlit so users can interact with MoodWise directly through the web.

---

## 🧠 What I Learned

As my first NLP project, MoodWise helped me understand:

* NLP fundamentals
* Text preprocessing
* Tokenization
* Stopword removal
* Bag of Words
* TF-IDF
* Multinomial Naive Bayes
* Logistic Regression
* Model comparison
* Train/test splitting
* Classification workflows
* Deploying an ML model with Streamlit

---

## 🔮 What's Next?

MoodWise is a starting point for exploring more advanced NLP techniques.

Future improvements could include:

* Hyperparameter tuning
* N-gram experimentation
* More detailed evaluation metrics
* Confusion matrix analysis
* Prediction confidence
* Deep-learning approaches
* Transformer-based NLP models

---

## 👨‍💻 Author

**Yash Vinay Kalyani**


`AI/ML` • `Data Science` • `Software Development`

---

⭐ **If you found MoodWise interesting, consider starring the repository!**
