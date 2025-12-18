# CSCIGA2433_Final_Project
database system management final project

Haotong Wu hw2933
Yujun Yao yy4107

This repository contains the final project for the CSCIGA2433 Database System Management course.



## 📂 Project Structure

```text
Part_4_V1.0/
├── app.py                          
├── insurance_pipeline_ml.py        
├── requirements.txt                
├── data/                          
│   └── medical_insurance_final.csv
├── saved_models/              
│   ├── premium_pipe.joblib         
│   ├── risk_model.joblib        
│   └── ...
└── templates/                    
    ├── index.html             
    ├── results.html             
    ├── admin.html          
    └── ...                     
```
## Installation & Setup
### 1. Prerequisites
Python 3.8 or higher
MongoDB Atlas Account (for cloud database)
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Environment Configuration
1. Create a `.env` file in the Part_4_v1.0 directory of the project.
2. Add your MongoDB connection string to it:
```plaintext
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?appName=Cluster0
```

## How to Run Part4
### 1.  Initialize ML Models
Before starting the web server, you must generate the machine learning model files. This script will train the models using the data in `data/medical_insurance_final.csv` and save them to the `saved_models/` folder.
```bash
python insurance_pipeline_ml.py
```
### 2. Start the Web Server
Run the Flask application:
```bash
python app.py
```
### 3. Access the Web Application
Open a web browser and navigate to `/http://127.0.0.1:5000` to access the web application.