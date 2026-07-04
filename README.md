# 🌱 Agri-Vision: Cotton Crop Analysis System

Agri-Vision is an AI-powered system that analyzes cotton crop images to determine growth stages and health conditions.  
It helps farmers and researchers make informed decisions about crop management and harvest timing.
<div align="center">

## 🌱 AI Crop Analysis Initialization

<img src="https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/Crop-Analysis-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Active-0ea5e9?style=for-the-badge"/>

</div>


> 🚀 This module prepares the complete AI-powered crop analysis workflow by handling image preprocessing, initializing prediction models, and validating uploaded crop images before analysis.

<br>

<table align="center">
<tr>
<td width="50%">

### ⚙️ Core Responsibilities
- 📸 Image preprocessing
- 🧠 AI model initialization
- ✅ Crop image validation
- 🔍 Prediction workflow handling
- ⚡ Optimized processing pipeline

</td>

<td width="50%">

### 🌟 Future Enhancements
- 🌾 Multi-crop disease detection
- 📡 Real-time prediction rendering
- ☁️ Cloud-based AI integration
- 📱 Mobile performance optimization
- 🤖 Advanced deep learning support

</td>
</tr>
</table>

---

<div align="center">


</div>

##  📌 Overview

Agri-Vision uses deep learning and computer vision techniques to:

- Detect cotton growth phases  
- Identify cotton crop diseases  
- Provide confidence scores and actionable recommendations  
- Offer both a web interface and a REST API  

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Dataset Information](#-dataset-information)
- [Model Information](#-model-information)
- [Model Performance & Benchmarking](docs/model-benchmarking.md)
- [Project Structure](#-project-structure)
- [Setup & Execution](#-setup--execution)
- [API Reference](#️-api-reference)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)

---


## ✨ Features

- 🌿 **Growth Phase Detection** (Supported for cotton and 🍅 Tomato)
- 💚 **Health Assessment** (disease & damage detection)
- 🤖 **AI-Powered Analysis** using deep learning
- 🌐 **Web Interface** (Flask-based)
- 📊 **REST API Support** for programmatic access
- 🎯 **Smart Recommendations** for farmers
- ⚡ **Fast Processing** (< 2 seconds per image) 

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-111827?style=flat-square&logo=yolo&logoColor=white)

> Built using modern AI, deep learning, and computer vision technologies for precision agriculture.

---

## Dataset Information
---
### For Cotton Crop
The datasets used for training the Growth Stage Prediction and Crop Disease Classification models were sourced from Roboflow.

### Growth Stage Prediction Dataset (For cotton crop)  

https://universe.roboflow.com/p-project-ebvkg/cotton-boll-growth-detection/dataset/5  

*The above dataset is also having appropriate labels for YOLO model training

### Crop Disease Classification Dataset (for cotton crop)  

https://universe.roboflow.com/deep-learning-nygzt/tomato-crop-diseases

## Growth Phases Detected

- Cotton Blossom
- Cotton Bud
- Early Boll
- Matured Cotton Boll
- Split Cotton Boll



## Health Issues Identified

- Healthy
- Aphids
- Army Worm
- Bacterial Blight
- Cotton Boll Rot
- Green Cotton Boll
- Powdery mildew
- Target spot

---
### For Tomato Crop
The datasets used for training the Growth Stage Prediction and Crop Disease Classification models were sourced from Roboflow.

### Growth Stage Prediction Dataset (For tomato crop)  

https://www.kaggle.com/datasets/arjunsudheer326/tomato-plant-stages-dataset


### Crop Disease Classification Dataset (for tomato crop)  

https://universe.roboflow.com/deep-learning-nygzt/tomato-crop-diseases
## Growth Phases Detected

- Early Vegetative
- Flowering initiation



## Health Issues Identified

- Early Blight
- Healthy
- Late blight
- Leaf miner
- Leaf mold
- Mosaic virus
- Septoria
- Spider mites
- Yellow leaf curl virus

---
## For Potato Crop
The datasets used for training the potato disease classification model were taken from kaggle

## Crop disease dataset (for potato)
https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset

## Health Issues Identified
- Early Blight
- Late Blight
- Healthy Leaf

## setup 
Download the dataset from the given URL and make sure to split the it into training data, testing data and validation data.
---

## Environment
The app requires a strong `SECRET_KEY` when running in production. This key signs session cookies and other secrets — keep it private.

To generate a key:

```
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Create a `.env` file in the project root and add at least:

```
SECRET_KEY=your-generated-secret
OPENWEATHER_API_KEY=your-openweather-key
```

Optional account lockout settings:

```env
ACCOUNT_LOCKOUT_ENABLED=true
MAX_FAILED_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15
ENABLE_SECURITY_AUDIT=true
```

Run the app in production mode locally:

```
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(64))')"
export FLASK_ENV=production
python -m flask run --host=0.0.0.0 --port=5000
```

During development the app will create a temporary key if `SECRET_KEY` is not set — do not use that value in production.

---
# 🤖 Model Information
<!-- --- -->
## For cotton crop
## Growth Stage Prediction Model
Model Used - YOLOv8   

Parameters - ~3M  

Layers - 73  


## Crop Disease Classification Model
Model Used - ResNet50  

Parameters - 25.6M

### Grad-CAM Explainability
Successful cotton disease classifications can include a Grad-CAM heatmap overlay generated from the final ResNet50 convolutional block (`layer4[-1]`). Generated visualizations are saved under `static/generated/gradcam/` and surfaced in the results page and API responses when available.


# 📊 Model Results
Check training curves and result snapshots inside the `results/` directory.
For confusion matrices, benchmark tables, and reproducibility notes, see [Model Performance & Benchmarking](docs/model-benchmarking.md).

## Metrics for YOLOv8 (Growth Stage Prediction)
mAP50 - 60.06%  

mAP95 - 34.8%  

R - 53.8%  

P - 62.7%  

Inference Time - 3.3ms  



## Metrics for ResNet50 (Cotton Crop Disease Classification)
Accuracy - 99.83%  

Precision - 99.83%  

Recall - 99.83%  

F1 Score - 99.83%  

ROC AUC - 99.98%  


---
## For tomato crop
## Growth Stage Prediction Model
Model Used - YOLOv8   

Parameters - ~3M  

Layers - 73  


## Crop Disease Classification Model
Model Used - ResNet50  

Parameters - 25.6M

# 📊 Model Results
Check training curves and result snapshots inside the `results/` directory.

## Metrics for YOLOv8 (Tomato crop disease prediction)
mAP50 - 95.4%  

mAP95 - 86.2%  

R - 88.5%  

P - 92.3%  

Inference Time - 1.3ms  



## Metrics for ResNet50 (Cotton Crop growth stage prediction)
Accuracy - 100%

Precision - 100% 

Recall - 100% 

F1 Score - 100%  

 
---


## 📁 Project Structure

```tree
Agri-Vision/
│
├── app/
│   ├── __init__.py
│   │
│   ├── routes/
│   │   ├── auth.py
│   │   ├── admin.py
│   │   ├── dashboard.py
│   │   ├── disease.py
│   │   ├── reports.py
│   │   ├── weather.py
│   │   └── yield_prediction.py
│   │
│   ├── services/
│   │   ├── disease_prediction_service.py
│   │   ├── recommendation_engine.py
│   │   ├── report_service.py
│   │   ├── weather_service.py
│   │   ├── yield_service.py
│   │   ├── image_quality.py
│   │   └── gradcam.py
│   │
│   ├── database/
│   │   └── models.py
│   │
│   ├── templates/
│   └── static/
│
├── ai_models/
│   ├── cotton/
│   ├── potato/
│   ├── tomato/
│   └── growth_stage/
│
├── training/
│   ├── notebooks/
│   │   ├── cotton_crop_disease_prediction.ipynb
│   │   ├── potato_crop_disease_classification.ipynb
│   │   ├── tomato_crop_disease_classification.ipynb
│   │   ├── tomato_growth_stages_classification.ipynb
│   │   └── cotton_growth_stage_prediction.ipynb
│   │
│   ├── train.py
│   ├── model_config.json
│   └── model_registry.py
│
├── database/
│   ├── create_admin.py
│   ├── add_sample_data.py
│   ├── populate_disease_data.py
│   └── populate_historical_data.py
│
├── tasks/
│   ├── celery_tasks.py
│   └── celery_worker.py
│
├── results/
│   ├── cotton/
│   ├── potato/
│   ├── tomato/
│   └── growth_stage/
│
├── docs/
│   ├── architecture.md
│   ├── MODEL_VERSIONING.md
│   ├── PDF_FEATURE_INTEGRATION.md
│   ├── PDF_IMPLEMENTATION_SUMMARY.md
│   ├── PDF_QUICK_START.md
│   ├── security.md
│   ├── model-benchmarking.md
│   └── api-documentation.md
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── runtime.txt
│
├── tools/
│   ├── check_quotes.py
│   ├── check_tags.py
│   ├── count_brackets.py
│   └── find_unmatched.py
│
├── tests/
│   ├── test_admin_auth.py
│   ├── test_app.py
│   ├── test_config.py
│   ├── test_explain.py
│   ├── test_recommendations.py
│   ├── test_weather.py
│   └── test_yield.py
│
├── client/
├── .github/
│
├── run.py
├── requirements.txt
├── requirements_minimal.txt
├── requirements_no_versions.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── Code_Of_Conduct.md
├── .env.example
├── .gitignore
└── pytest.ini
```

---

# 🚀 Setup & Execution

There are two ways to run this project: using Docker (Recommended) or setting it up locally with Python.

## 🐳 Option A: Run with Docker (Recommended)
Using Docker is the easiest way to run Agri-Vision as it avoids system dependency issues and automatically sets up the environment.

1. Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
2. Clone the repository and navigate into it:
   ```bash
  git clone <https://github.com/neeru24/Agri-Vision>
  cd <Agri-Vision>
  ### Create Virtual Environment

```bash
python -m venv venv


Be careful with the markdown formatting/backticks.


### Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```
   ```

   ### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Flask App

```bash
python app.py
```
### Open in Browser

```txt
http://127.0.0.1:5000/
```

3. Build and start the container:
   ```bash
   docker-compose up --build
   ```
4. Access the web interface at `http://localhost:5000`.

---

## 🐍 Option B: Local Python Setup

If you prefer to run the project natively using Python (requires Python 3.8+):

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/neeru24/Agri-Vision.git
cd Agri-Vision
```

### 2️⃣ Create and Activate a Virtual Environment

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


### 3️⃣ Create a `.env` File

Create a `.env` file in the project root and add a `SECRET_KEY` entry.

This value is required in production—the app will not start without it. To generate a secure key locally, run:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(64))'
```

Then add the generated value to `.env`:

```env
SECRET_KEY=your_generated_secret_here
```

Optional account lockout settings:

```env
ACCOUNT_LOCKOUT_ENABLED=true
MAX_FAILED_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=15
ENABLE_SECURITY_AUDIT=true
```

### 4️⃣ Install Python Dependencies

Install all the required Python packages using:

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Project

Start the application explicitly by running:

```bash
python app.py
```

### ✅ Setup Complete

The project should now be running successfully on your local machine at `http://localhost:5000`.

---

## 🧪 Running Unit Tests & Coverage

Agri-Vision includes a comprehensive unit and integration testing suite built using `pytest` and `pytest-cov`. 

The test suite runs programmatically in-memory, requiring no external files or slow deep learning model loading. This ensures tests run in **less than 1 second** with **89%+ code coverage**.

### 1️⃣ Run Unit Tests & Coverage (Simultaneously)

Thanks to the pre-configured `pytest.ini`, you don't need to pass long command line arguments. Running a single command will execute all 28 tests, generate verbose progress, check code coverage, and produce an HTML report automatically:

```bash
python -m pytest
```

**Expected Output:**
```text
tests/test_app.py::test_preprocess_image_for_resnet PASSED               [  3%]
tests/test_app.py::test_infer_disease_fallback PASSED                    [  7%]
...
tests/test_app.py::test_post_api_analyze_exception PASSED                [100%]

=============================== tests coverage ================================
Name      Stmts   Miss  Cover   Missing
--------------------------------------
app.py     201     22    89%   81-83, 88-90, 159, 406-420
--------------------------------------
TOTAL      201     22    89%

Coverage HTML written to dir htmlcov
============================= 28 passed in 1.06s ==============================
```

### 2️⃣ View Interactive HTML Coverage report

When you run the tests, a beautiful interactive HTML coverage report is automatically created in the `htmlcov/` directory.

To visually inspect which lines are covered (in green) and which are missed (in red) line-by-line:
1. Open the folder `htmlcov/` in your file explorer.
2. Double-click `index.html` to open it in any web browser.
3. Click on `app.py` to view the beautiful interactive code visualization.

### 3️⃣ Automated Continuous Integration (CI)

A GitHub Actions workflow is fully set up. It will automatically run your entire unit test suite and verify code quality/coverage metrics on every single `push` or `pull_request` to the `main` branch.

---



# 🛠️ API Reference

## Analyze Image (POST Request)

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@cotton_image.jpg" \
  -F "lat=30.0444" \
  -F "lon=31.2357" \
  -F "field_acres=5"
```

---

## 📦 Response Format (JSON)

```json
{
  "status": "success",
  "timestamp": "2026-06-03T12:00:00.123456",
  "weather": {
    "temperature": 32,
    "humidity": 55,
    "precipitation": 0
  },
  "results": {
    "disease": { "predicted_class": "Healthy", "health_score": 82.0 },
    "growth": { "main_class": "Matured Cotton Boll" },
    "recommendations": ["..."],
    "yield_estimate": {
      "weather_multiplier": 1.0,
      "weather_notes": ["Weather conditions are favourable for cotton."],
      "combined_multiplier": 0.612,
      "stage_multiplier": 0.85,
      "health_multiplier": 0.72
    }
  }
}
```

---

# 🎯 Usage

## 🌐 Web Interface

1. Go to `/analyze`
2. Upload a cotton crop image
3. View detailed analysis results
4. Download the JSON report if needed

---



# 🚀 Future Enhancements

- 📱 Mobile application support  
- 🎥 Real-time video analysis  
- 🌾 Multi-crop support (Cotton, Tomato, and Potato fully integrated)
- ☁️ Weather data integration  
- 📊 Yield prediction system  
- 🧠 Improved AI models
---

# 🤝 Contributing

Contributions are welcome to improve Agri-Vision and make it more useful for farmers, researchers, and developers.

Feel free to:

- Fork the repository  
- Create a feature branch  
- Submit a pull request  

## Contributors

<!-- CONTRIBUTORS_START -->
<a href="https://github.com/4nshhh"><img src="https://github.com/4nshhh.png" width="50px" loading="lazy" title="4nshhh" style="border-radius:50%;margin:5px;" alt="4nshhh" /></a><a href="https://github.com/ANSHIKATYAGI30"><img src="https://github.com/ANSHIKATYAGI30.png" width="50px" loading="lazy" title="ANSHIKATYAGI30" style="border-radius:50%;margin:5px;" alt="ANSHIKATYAGI30" /></a><a href="https://github.com/AanyaJain0811"><img src="https://github.com/AanyaJain0811.png" width="50px" loading="lazy" title="AanyaJain0811" style="border-radius:50%;margin:5px;" alt="AanyaJain0811" /></a><a href="https://github.com/Akshu121796"><img src="https://github.com/Akshu121796.png" width="50px" loading="lazy" title="Akshu121796" style="border-radius:50%;margin:5px;" alt="Akshu121796" /></a><a href="https://github.com/Anandsirigiri07"><img src="https://github.com/Anandsirigiri07.png" width="50px" loading="lazy" title="Anandsirigiri07" style="border-radius:50%;margin:5px;" alt="Anandsirigiri07" /></a><a href="https://github.com/Anshuman-cs50"><img src="https://github.com/Anshuman-cs50.png" width="50px" loading="lazy" title="Anshuman-cs50" style="border-radius:50%;margin:5px;" alt="Anshuman-cs50" /></a><a href="https://github.com/Anvitha-Samineni"><img src="https://github.com/Anvitha-Samineni.png" width="50px" loading="lazy" title="Anvitha-Samineni" style="border-radius:50%;margin:5px;" alt="Anvitha-Samineni" /></a><a href="https://github.com/Bhagy-Yelleti"><img src="https://github.com/Bhagy-Yelleti.png" width="50px" loading="lazy" title="Bhagy-Yelleti" style="border-radius:50%;margin:5px;" alt="Bhagy-Yelleti" /></a><a href="https://github.com/CyberHunter12-ui"><img src="https://github.com/CyberHunter12-ui.png" width="50px" loading="lazy" title="CyberHunter12-ui" style="border-radius:50%;margin:5px;" alt="CyberHunter12-ui" /></a><a href="https://github.com/Harshan07-web"><img src="https://github.com/Harshan07-web.png" width="50px" loading="lazy" title="Harshan07-web" style="border-radius:50%;margin:5px;" alt="Harshan07-web" /></a><a href="https://github.com/KD2303"><img src="https://github.com/KD2303.png" width="50px" loading="lazy" title="KD2303" style="border-radius:50%;margin:5px;" alt="KD2303" /></a><a href="https://github.com/Kuki-09"><img src="https://github.com/Kuki-09.png" width="50px" loading="lazy" title="Kuki-09" style="border-radius:50%;margin:5px;" alt="Kuki-09" /></a><a href="https://github.com/Kunal241207"><img src="https://github.com/Kunal241207.png" width="50px" loading="lazy" title="Kunal241207" style="border-radius:50%;margin:5px;" alt="Kunal241207" /></a><a href="https://github.com/Lakshpreetkaur"><img src="https://github.com/Lakshpreetkaur.png" width="50px" loading="lazy" title="Lakshpreetkaur" style="border-radius:50%;margin:5px;" alt="Lakshpreetkaur" /></a><a href="https://github.com/Manpreet661"><img src="https://github.com/Manpreet661.png" width="50px" loading="lazy" title="Manpreet661" style="border-radius:50%;margin:5px;" alt="Manpreet661" /></a><a href="https://github.com/MansiSaini14"><img src="https://github.com/MansiSaini14.png" width="50px" loading="lazy" title="MansiSaini14" style="border-radius:50%;margin:5px;" alt="MansiSaini14" /></a><a href="https://github.com/NavanChakravarthiHS"><img src="https://github.com/NavanChakravarthiHS.png" width="50px" loading="lazy" title="NavanChakravarthiHS" style="border-radius:50%;margin:5px;" alt="NavanChakravarthiHS" /></a><a href="https://github.com/Nehachavan03"><img src="https://github.com/Nehachavan03.png" width="50px" loading="lazy" title="Nehachavan03" style="border-radius:50%;margin:5px;" alt="Nehachavan03" /></a><a href="https://github.com/Nidhisharora"><img src="https://github.com/Nidhisharora.png" width="50px" loading="lazy" title="Nidhisharora" style="border-radius:50%;margin:5px;" alt="Nidhisharora" /></a><a href="https://github.com/Nidzz07"><img src="https://github.com/Nidzz07.png" width="50px" loading="lazy" title="Nidzz07" style="border-radius:50%;margin:5px;" alt="Nidzz07" /></a><a href="https://github.com/NishiSingh04"><img src="https://github.com/NishiSingh04.png" width="50px" loading="lazy" title="NishiSingh04" style="border-radius:50%;margin:5px;" alt="NishiSingh04" /></a><a href="https://github.com/Nishita-Thakur"><img src="https://github.com/Nishita-Thakur.png" width="50px" loading="lazy" title="Nishita-Thakur" style="border-radius:50%;margin:5px;" alt="Nishita-Thakur" /></a><a href="https://github.com/PalashKulkarni"><img src="https://github.com/PalashKulkarni.png" width="50px" loading="lazy" title="PalashKulkarni" style="border-radius:50%;margin:5px;" alt="PalashKulkarni" /></a><a href="https://github.com/PranshuPujara"><img src="https://github.com/PranshuPujara.png" width="50px" loading="lazy" title="PranshuPujara" style="border-radius:50%;margin:5px;" alt="PranshuPujara" /></a><a href="https://github.com/Pranxxth-D"><img src="https://github.com/Pranxxth-D.png" width="50px" loading="lazy" title="Pranxxth-D" style="border-radius:50%;margin:5px;" alt="Pranxxth-D" /></a><a href="https://github.com/PremSahith"><img src="https://github.com/PremSahith.png" width="50px" loading="lazy" title="PremSahith" style="border-radius:50%;margin:5px;" alt="PremSahith" /></a><a href="https://github.com/Sarvesh-web2"><img src="https://github.com/Sarvesh-web2.png" width="50px" loading="lazy" title="Sarvesh-web2" style="border-radius:50%;margin:5px;" alt="Sarvesh-web2" /></a><a href="https://github.com/Secret371"><img src="https://github.com/Secret371.png" width="50px" loading="lazy" title="Secret371" style="border-radius:50%;margin:5px;" alt="Secret371" /></a><a href="https://github.com/Sh8ubham"><img src="https://github.com/Sh8ubham.png" width="50px" loading="lazy" title="Sh8ubham" style="border-radius:50%;margin:5px;" alt="Sh8ubham" /></a><a href="https://github.com/ShreyasPatil3105"><img src="https://github.com/ShreyasPatil3105.png" width="50px" loading="lazy" title="ShreyasPatil3105" style="border-radius:50%;margin:5px;" alt="ShreyasPatil3105" /></a><a href="https://github.com/Snnehamaurya"><img src="https://github.com/Snnehamaurya.png" width="50px" loading="lazy" title="Snnehamaurya" style="border-radius:50%;margin:5px;" alt="Snnehamaurya" /></a><a href="https://github.com/SujalMahapatra"><img src="https://github.com/SujalMahapatra.png" width="50px" loading="lazy" title="SujalMahapatra" style="border-radius:50%;margin:5px;" alt="SujalMahapatra" /></a><a href="https://github.com/Sukhmanpreetkaur18"><img src="https://github.com/Sukhmanpreetkaur18.png" width="50px" loading="lazy" title="Sukhmanpreetkaur18" style="border-radius:50%;margin:5px;" alt="Sukhmanpreetkaur18" /></a><a href="https://github.com/Suyash2527"><img src="https://github.com/Suyash2527.png" width="50px" loading="lazy" title="Suyash2527" style="border-radius:50%;margin:5px;" alt="Suyash2527" /></a><a href="https://github.com/VINAY-KUMAR855"><img src="https://github.com/VINAY-KUMAR855.png" width="50px" loading="lazy" title="VINAY-KUMAR855" style="border-radius:50%;margin:5px;" alt="VINAY-KUMAR855" /></a><a href="https://github.com/VishnuPriya110792"><img src="https://github.com/VishnuPriya110792.png" width="50px" loading="lazy" title="VishnuPriya110792" style="border-radius:50%;margin:5px;" alt="VishnuPriya110792" /></a><a href="https://github.com/Xploit-Ghost"><img src="https://github.com/Xploit-Ghost.png" width="50px" loading="lazy" title="Xploit-Ghost" style="border-radius:50%;margin:5px;" alt="Xploit-Ghost" /></a><a href="https://github.com/abhinavkdeval08-design"><img src="https://github.com/abhinavkdeval08-design.png" width="50px" loading="lazy" title="abhinavkdeval08-design" style="border-radius:50%;margin:5px;" alt="abhinavkdeval08-design" /></a><a href="https://github.com/anshul23102"><img src="https://github.com/anshul23102.png" width="50px" loading="lazy" title="anshul23102" style="border-radius:50%;margin:5px;" alt="anshul23102" /></a><a href="https://github.com/anujaiitj123"><img src="https://github.com/anujaiitj123.png" width="50px" loading="lazy" title="anujaiitj123" style="border-radius:50%;margin:5px;" alt="anujaiitj123" /></a><a href="https://github.com/anushka11p"><img src="https://github.com/anushka11p.png" width="50px" loading="lazy" title="anushka11p" style="border-radius:50%;margin:5px;" alt="anushka11p" /></a><a href="https://github.com/bhavishyaverma450"><img src="https://github.com/bhavishyaverma450.png" width="50px" loading="lazy" title="bhavishyaverma450" style="border-radius:50%;margin:5px;" alt="bhavishyaverma450" /></a><a href="https://github.com/cosmoqain459"><img src="https://github.com/cosmoqain459.png" width="50px" loading="lazy" title="cosmoqain459" style="border-radius:50%;margin:5px;" alt="cosmoqain459" /></a><a href="https://github.com/darshil2032007"><img src="https://github.com/darshil2032007.png" width="50px" loading="lazy" title="darshil2032007" style="border-radius:50%;margin:5px;" alt="darshil2032007" /></a><a href="https://github.com/gourikataneja"><img src="https://github.com/gourikataneja.png" width="50px" loading="lazy" title="gourikataneja" style="border-radius:50%;margin:5px;" alt="gourikataneja" /></a><a href="https://github.com/grishabhatia"><img src="https://github.com/grishabhatia.png" width="50px" loading="lazy" title="grishabhatia" style="border-radius:50%;margin:5px;" alt="grishabhatia" /></a><a href="https://github.com/hetvi1422"><img src="https://github.com/hetvi1422.png" width="50px" loading="lazy" title="hetvi1422" style="border-radius:50%;margin:5px;" alt="hetvi1422" /></a><a href="https://github.com/iharmandeepsingh"><img src="https://github.com/iharmandeepsingh.png" width="50px" loading="lazy" title="iharmandeepsingh" style="border-radius:50%;margin:5px;" alt="iharmandeepsingh" /></a><a href="https://github.com/itsdakshjain"><img src="https://github.com/itsdakshjain.png" width="50px" loading="lazy" title="itsdakshjain" style="border-radius:50%;margin:5px;" alt="itsdakshjain" /></a><a href="https://github.com/jwalkorat"><img src="https://github.com/jwalkorat.png" width="50px" loading="lazy" title="jwalkorat" style="border-radius:50%;margin:5px;" alt="jwalkorat" /></a><a href="https://github.com/kashviporwal-byte"><img src="https://github.com/kashviporwal-byte.png" width="50px" loading="lazy" title="kashviporwal-byte" style="border-radius:50%;margin:5px;" alt="kashviporwal-byte" /></a><a href="https://github.com/kesavvvvvv"><img src="https://github.com/kesavvvvvv.png" width="50px" loading="lazy" title="kesavvvvvv" style="border-radius:50%;margin:5px;" alt="kesavvvvvv" /></a><a href="https://github.com/knoxiboy"><img src="https://github.com/knoxiboy.png" width="50px" loading="lazy" title="knoxiboy" style="border-radius:50%;margin:5px;" alt="knoxiboy" /></a><a href="https://github.com/krishkhinchi"><img src="https://github.com/krishkhinchi.png" width="50px" loading="lazy" title="krishkhinchi" style="border-radius:50%;margin:5px;" alt="krishkhinchi" /></a><a href="https://github.com/krushnanirmalkar"><img src="https://github.com/krushnanirmalkar.png" width="50px" loading="lazy" title="krushnanirmalkar" style="border-radius:50%;margin:5px;" alt="krushnanirmalkar" /></a><a href="https://github.com/ksrikarsai"><img src="https://github.com/ksrikarsai.png" width="50px" loading="lazy" title="ksrikarsai" style="border-radius:50%;margin:5px;" alt="ksrikarsai" /></a><a href="https://github.com/kunal-9090"><img src="https://github.com/kunal-9090.png" width="50px" loading="lazy" title="kunal-9090" style="border-radius:50%;margin:5px;" alt="kunal-9090" /></a><a href="https://github.com/manaswi3"><img src="https://github.com/manaswi3.png" width="50px" loading="lazy" title="manaswi3" style="border-radius:50%;margin:5px;" alt="manaswi3" /></a><a href="https://github.com/neeru24"><img src="https://github.com/neeru24.png" width="50px" loading="lazy" title="neeru24" style="border-radius:50%;margin:5px;" alt="neeru24" /></a><a href="https://github.com/nehala5"><img src="https://github.com/nehala5.png" width="50px" loading="lazy" title="nehala5" style="border-radius:50%;margin:5px;" alt="nehala5" /></a><a href="https://github.com/om-bhinsara"><img src="https://github.com/om-bhinsara.png" width="50px" loading="lazy" title="om-bhinsara" style="border-radius:50%;margin:5px;" alt="om-bhinsara" /></a><a href="https://github.com/pranavshankar1221"><img src="https://github.com/pranavshankar1221.png" width="50px" loading="lazy" title="pranavshankar1221" style="border-radius:50%;margin:5px;" alt="pranavshankar1221" /></a><a href="https://github.com/princejain-2004"><img src="https://github.com/princejain-2004.png" width="50px" loading="lazy" title="princejain-2004" style="border-radius:50%;margin:5px;" alt="princejain-2004" /></a><a href="https://github.com/rachel-d-07"><img src="https://github.com/rachel-d-07.png" width="50px" loading="lazy" title="rachel-d-07" style="border-radius:50%;margin:5px;" alt="rachel-d-07" /></a><a href="https://github.com/ravikumar-prajapati-14"><img src="https://github.com/ravikumar-prajapati-14.png" width="50px" loading="lazy" title="ravikumar-prajapati-14" style="border-radius:50%;margin:5px;" alt="ravikumar-prajapati-14" /></a><a href="https://github.com/rishabh0510rishabh"><img src="https://github.com/rishabh0510rishabh.png" width="50px" loading="lazy" title="rishabh0510rishabh" style="border-radius:50%;margin:5px;" alt="rishabh0510rishabh" /></a><a href="https://github.com/rishika526"><img src="https://github.com/rishika526.png" width="50px" loading="lazy" title="rishika526" style="border-radius:50%;margin:5px;" alt="rishika526" /></a><a href="https://github.com/ryanzone"><img src="https://github.com/ryanzone.png" width="50px" loading="lazy" title="ryanzone" style="border-radius:50%;margin:5px;" alt="ryanzone" /></a><a href="https://github.com/sahare-mayur-0071"><img src="https://github.com/sahare-mayur-0071.png" width="50px" loading="lazy" title="sahare-mayur-0071" style="border-radius:50%;margin:5px;" alt="sahare-mayur-0071" /></a><a href="https://github.com/samrin1502"><img src="https://github.com/samrin1502.png" width="50px" loading="lazy" title="samrin1502" style="border-radius:50%;margin:5px;" alt="samrin1502" /></a><a href="https://github.com/sanikayadav2024"><img src="https://github.com/sanikayadav2024.png" width="50px" loading="lazy" title="sanikayadav2024" style="border-radius:50%;margin:5px;" alt="sanikayadav2024" /></a><a href="https://github.com/saurabhhhcodes"><img src="https://github.com/saurabhhhcodes.png" width="50px" loading="lazy" title="saurabhhhcodes" style="border-radius:50%;margin:5px;" alt="saurabhhhcodes" /></a><a href="https://github.com/shrey2597"><img src="https://github.com/shrey2597.png" width="50px" loading="lazy" title="shrey2597" style="border-radius:50%;margin:5px;" alt="shrey2597" /></a><a href="https://github.com/shreyansh-tech21"><img src="https://github.com/shreyansh-tech21.png" width="50px" loading="lazy" title="shreyansh-tech21" style="border-radius:50%;margin:5px;" alt="shreyansh-tech21" /></a><a href="https://github.com/siddharth277"><img src="https://github.com/siddharth277.png" width="50px" loading="lazy" title="siddharth277" style="border-radius:50%;margin:5px;" alt="siddharth277" /></a><a href="https://github.com/suryansh24-coder"><img src="https://github.com/suryansh24-coder.png" width="50px" loading="lazy" title="suryansh24-coder" style="border-radius:50%;margin:5px;" alt="suryansh24-coder" /></a><a href="https://github.com/upasana-2006"><img src="https://github.com/upasana-2006.png" width="50px" loading="lazy" title="upasana-2006" style="border-radius:50%;margin:5px;" alt="upasana-2006" /></a><a href="https://github.com/varsha-2503"><img src="https://github.com/varsha-2503.png" width="50px" loading="lazy" title="varsha-2503" style="border-radius:50%;margin:5px;" alt="varsha-2503" /></a><a href="https://github.com/varshini-nandula"><img src="https://github.com/varshini-nandula.png" width="50px" loading="lazy" title="varshini-nandula" style="border-radius:50%;margin:5px;" alt="varshini-nandula" /></a><a href="https://github.com/vd77-1"><img src="https://github.com/vd77-1.png" width="50px" loading="lazy" title="vd77-1" style="border-radius:50%;margin:5px;" alt="vd77-1" /></a><a href="https://github.com/vedikabajaj05"><img src="https://github.com/vedikabajaj05.png" width="50px" loading="lazy" title="vedikabajaj05" style="border-radius:50%;margin:5px;" alt="vedikabajaj05" /></a><a href="https://github.com/yeshitakondasani-1237"><img src="https://github.com/yeshitakondasani-1237.png" width="50px" loading="lazy" title="yeshitakondasani-1237" style="border-radius:50%;margin:5px;" alt="yeshitakondasani-1237" /></a>
<!-- CONTRIBUTORS_END -->

---

## 📝 Additional Notes

- Follow the project structure and coding style.
- Avoid spam or duplicate PRs/issues.
- Be respectful during code reviews and discussions.
- Beginners are welcome — feel free to ask questions if stuck ✨

---

# 📜 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.

---

# 🙌 Acknowledgements

Special thanks to:

- TensorFlow  
- Flask  
- OpenCV  
- Open-source contributors  
- Agricultural research datasets  
- Ultralytics
- PyTorch

---

<div align="center">

## ❤️ Made with Passion by [neeru24](https://github.com/neeru24)

⭐ If you found this project helpful, consider giving it a star. ⭐

</div>
