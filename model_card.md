# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## **Model Details**

- **Model Type:** Random Forest Classifier
- **Framework:** Scikit-learn
- **Purpose:** Predict whether an individual's income exceeds $50K per year based on census data.
- **Features Used:**
  - **Numerical Features:** 
    - `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
  - **Categorical Features:**
    - `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `native-country`
  - **Binary Features:** 
    - `sex`, `salary`
- **Trained On:** UCI Census Income dataset ([Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)).

## **Intended Use**

- **Primary Users:** Researchers and data scientists working on income classification and demographic analysis.
- **Use Case:** Classifying individuals as earning more or less than $50K annually.
- **Limitations:** The model is trained on data from the 1994 US Census and may not generalize to other populations, geographic areas, or time periods.

## **Training Data**

- **Source:** UCI Census Income dataset
- **Size:** 48,842 rows, 14 features.
- **Cleaning Steps:**
  - Removed whitespace from categorical features.
  - Encoded categorical features with one-hot encoding.
  - Converted binary targets into integers.
  - Split into train-test with 80-20 ratio.
- **Distribution:**
  - Binary Target Classes: 
    - ```>50K: 24%```
    - ```<=50K: 76%```

## **Evaluation Data**

- **Source:** 20% of the UCI Census dataset (test split).
- **Cleaning and Preprocessing:** Same pipeline as training data.
- **Size:** 9,768 rows, 14 features.

## **Metrics**

Metrics are computed for both overall and sliced datasets:

### **Overall Performance:**

- **Precision:** 95.6%
- **Recall:** 92.45%
- **FBeta:** 94.0%

### **Sliced Performance:**
Performance metrics were computed for slices of the dataset based on categorical feature values. Below are some examples:

#### **Education:**
- **Bachelors:** Precision: 96.83%, Recall: 94.89%, F1: 95.85%
- **Masters:** Precision: 98.0%, Recall: 98.0%, F1: 98.0%
- **HS-grad:** Precision: 94.68%, Recall: 86.89%, F1: 90.62%

#### **Workclass:**
- **Private:** Precision: 96.91%, Recall: 92.02%, F1: 94.40%
- **Self-emp-not-inc:** Precision: 95.2%, Recall: 92.97%, F1: 94.07%

#### **Race:**
- **White:** Precision: 96.29%, Recall: 92.72%, F1: 94.47%
- **Black:** Precision: 98.55%, Recall: 87.18%, F1: 92.52%

## **Ethical Considerations**

- **Bias in Data:** 
  - The dataset reflects demographic and income patterns from 1994, which may perpetuate historical biases, especially in features such as `workclass`, `education`, and `race`.
  - For example, certain racial groups like "White" show higher recall and precision compared to "Amer-Indian-Eskimo."
- **Fairness Issues:** 
  - Potentially biased predictions for underrepresented groups, such as certain categories in `native-country` or `race`.
  - Imbalances in income class distribution (76% earn <=50K).
- **Data Privacy:** 
  - The dataset does not include sensitive information but may raise privacy concerns when used on live data.

## **Caveats and Recommendations**

- **Dataset Limitations:**
The model reflects the economic and social structure of the US in 1994 and may not generalize to modern or international settings.
Missing values for features like `workclass` and `occupation` could lead to reduced model reliability.
- **Performance on Slices:** 
  Some slices (e.g., `Vietnam` in `native-country`) show extreme recall values (e.g., 100%) due to limited representation in the dataset.
- **Recommendations:**
  - Validate and fine-tune the model on contemporary data before deployment.
  - Incorporate more balanced datasets to mitigate biases.
