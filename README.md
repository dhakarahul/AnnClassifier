
## **Project Overview**  
This project aims to predict **customer churn** using an **Artificial Neural Network (ANN) classifier**. The model is trained on customer data and predicts whether a customer is likely to leave the system.  

The project is implemented using **Streamlit**, allowing users to input customer details and get real-time churn predictions.  

---

## **Tech Stack**  
- **Frontend:** Streamlit  
- **Backend:** TensorFlow (Keras), Scikit-Learn, Pandas, NumPy  
- **Model Type:** Artificial Neural Network (ANN)  
- **Encoders:**  
  - One-Hot Encoding
  - Label Encoding
  - Standard Scaling for numerical features  
- **Model Deployment:** Local execution with Streamlit  

---

## **Dataset & Features**  
The model was trained on a dataset containing customer demographics and financial information.  

### **Features Used**  
| Feature | Description |
|---------|------------|
| `CreditScore` | Credit score of the customer |
| `Gender` | Male or Female (encoded) |
| `Age` | Age of the customer |
| `Tenure` | Number of years the customer has been with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card (0 or 1) |
| `IsActiveMember` | Whether the customer is an active member (0 or 1) |
| `EstimatedSalary` | Estimated salary of the customer |
| `Geography` | Customer's country (One-Hot Encoded: France, Germany, Spain) |

---

## **Installation & Setup**  
### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/your-repo/customer-churn-ann.git
cd customer-churn-ann
```

### **Step 2: Create a Virtual Environment & Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Streamlit App**  
```bash
streamlit run app.py
```

---

## **Model Pipeline**  
1. **Load Pre-trained Models & Scalers**
   - ANN model (`model.h5`)
   - Encoders (`one_hot_encoder_geo.pkl`, `label_encoder_gender.pkl`)
   - Standard Scaler (`scaler.pkl`)
   
2. **Take User Input via Streamlit UI**
   - Users enter details like credit score, age, tenure, balance, etc.
   
3. **Preprocess Input**
   - Encode categorical variables (`Geography`, `Gender`)
   - Scale numerical features

4. **Make Prediction**
   - The ANN model predicts churn probability
   - If probability **> 0.5**, customer is **likely to churn**
   - Else, customer is **not likely to churn**

5. **Display Result in Streamlit**
   - "Customer is going to churn" or "Customer is not going to churn"

---

## **Example Prediction**  
| Input Feature | Value |
|--------------|-------|
| CreditScore | 650 |
| Gender | Male |
| Age | 35 |
| Tenure | 5 |
| Balance | 50,000 |
| NumOfProducts | 2 |
| HasCrCard | 1 |
| IsActiveMember | 1 |
| EstimatedSalary | 70,000 |
| Geography | France |

🔹 **Prediction Output:** _"Customer is not going to churn"_  

---

## **Troubleshooting & Common Errors**  
### **1. ValueError: Feature Names Do Not Match**  
🔹 **Solution:** Ensure feature names match exactly before applying `scaler.transform()`.  
```python
expected_columns = scaler.feature_names_in_
input_data = input_data[expected_columns]
```

### **2. Geography Encoding Error**  
🔹 **Solution:** Use `.transform()` instead of `.fit_transform()` and pass the correct format.  
```python
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
```
---

## **Contributors**  
Rahul Dhaka, Data Scientist
Email = dhakarahul.176@gmail.com 
