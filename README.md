# 🌱 Agri-Vision: Cotton Crop Analysis System

Agri-Vision is an AI-powered system that analyzes cotton crop images to determine growth stages and health conditions.  
It helps farmers and researchers make informed decisions about crop management and harvest timing.

---

## 📌 Overview

Agri-Vision uses deep learning and computer vision techniques to:

- Detect cotton growth phases  
- Identify cotton crop diseases  
- Provide confidence scores and actionable recommendations  
- Offer both a web interface and a REST API  

---

## ✨ Features

- 🌿 **Growth Phase Detection** (4 stages for cotton, 🍅 Tomato coming soon)
- 💚 **Health Assessment** (disease & damage detection)
- 🤖 **AI-Powered Analysis** using deep learning
- 🌐 **Web Interface** (Flask-based)
- 📊 **REST API Support** for programmatic access
- 🎯 **Smart Recommendations** for farmers
- ⚡ **Fast Processing** (< 2 seconds per image) 

---

## 🛠️ Tech Stack

- Python  
- Flask  
- TensorFlow  
- OpenCV  
- HTML/CSS  
- JavaScript  
- Ultralytics
- Pytorch

---

## Dataset Information
Both the datasets related to the model training of Growth stage prediction model, crop disease classification model were taken from roboflow

Link to the dataset used for training growth stage prediction model- https://universe.roboflow.com/p-project-ebvkg/cotton-boll-growth-detection/dataset/5
*The above dataset is also having appropriate labels for YOLO model training

Link to the dataset used for training crop disease classification model- https://universe.roboflow.com/deep-learning-nygzt/cotton-crop-dieases/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

## 🔜 Upcoming: Tomato Crop Support

We are actively working on expanding Agri-Vision to support **Tomato crop analysis**. Planned features include:

- 🍅 Tomato disease detection (Early Blight, Late Blight, Septoria Leaf Spot, Tomato Mosaic Virus)
- 🌱 Tomato growth stage identification (Seedling, Vegetative, Flowering, Fruit development)
- 📊 Tomato-specific recommendations

**Dataset references for contributors:**
- [PlantVillage Tomato Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
- [Tomato Disease Detection (Roboflow)](https://universe.roboflow.com/search?q=tomato)

> 💡 **Contributors wanted!** If you're interested in helping build tomato crop models, check out issue [#73](https://github.com/neeru24/Agri-Vision/issues/73)

## Growth Phases Detected

- Cotton Blossom
- Cotton Bud
- Early Boll
- Matured Cotton Boll
- Split Cotton Boll

---

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

# Model Information
## Growth Stage Prediction Model
Model used- YOLOv8
Parameters- ~3M
Layers- 73

## Crop Disease Classification Model
Model Used- Resnet50
Parameters- 25.6M


# Model Results
checkout snapshots, training curves at results/
## Metrics for YOLOv8(Growth stage prediction)
mAP50- 60.06%
mAP95- 34.8%
R- 53.8%
P- 62.7%
Inference Time- 3.3ms


## Metrics for Resnet50(Cotton crop disease classification)
Accuracy- 99.83%
Precision- 99.83%
Recall- 99.83%
F1 Score- 99.83%
ROC AUC- 99.98%


---


## 📁 Project Structure

```tree
Agri-Vision/
│
├── results/                        # Stores output results and visualizations
│   └── training_history.png
│
├── static/                         # Static assets
│   ├── css/
│   ├── uploads/
│   └── favicon.png
│
├── templates/                      # Flask HTML templates
│   ├── index.html
│   ├── results.html
│   └── upload.html
│
├── .env                            # Environment variables (create manually)
├── .gitignore
├── app.py                          # Main Flask application
├── LICENSE
├── README.md
├── requirements.txt
└── train.py                        # Model training script
```

---

# 🚀 Project Setup & Execution

Follow the steps below to run the project locally.

## 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

---

## 2️⃣ Create a `.env` File

Create a `.env` file in the root directory of the project and add your secret key.

### Example

```env
SECRET_KEY=your_secret_key_here
```

Replace `your_secret_key_here` with your actual secret key.

---

## 3️⃣ Install Python Dependencies

Install all the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Project

Start the application using:

```bash
python app.py
```

---

## ✅ Setup Complete

The project should now be running successfully on your local machine.

---

## 🧪 Running Unit Tests & Coverage

Agri-Vision includes a comprehensive unit and integration testing suite built using `pytest` and `pytest-cov`. 

The test suite runs programmatically in-memory, requiring no external files or slow deep learning model loading. This ensures tests run in **less than 1 second** with **89%+ code coverage**.

### 1️⃣ Run Unit Tests & Coverage (Simultaneously)

Thanks to the pre-configured [pytest.ini](file:///d:/GITHUB%20LOCAL/Agri-Vision/pytest.ini), you don't need to pass long command line arguments. Running a single command will execute all 28 tests, generate verbose progress, check code coverage, and produce an HTML report automatically:

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
Name     Stmts   Miss  Cover   Missing
--------------------------------------
app.py     201     22    89%   81-83, 88-90, 159, 406-420
--------------------------------------
TOTAL      201     22    89%

Coverage HTML written to dir htmlcov
============================= 28 passed in 1.06s ==============================
```

### 2️⃣ View Interactive HTML Coverage Report

When you run the tests, a beautiful interactive HTML coverage report is automatically created in the `htmlcov/` directory.

To visually inspect which lines are covered (in green) and which are missed (in red) line-by-line:
1. Open the folder `htmlcov/` in your file explorer.
2. Double-click `index.html` to open it in any web browser.
3. Click on `app.py` to view the beautiful interactive code visualization.

### 3️⃣ Automated Continuous Integration (CI)

A GitHub Actions workflow is fully set up in [.github/workflows/tests.yml](file:///d:/GITHUB%20LOCAL/Agri-Vision/.github/workflows/tests.yml). It will automatically run your entire unit test suite and verify code quality/coverage metrics on every single `push` or `pull_request` to the `main` branch.

---



# 🛠️ API Reference

## Analyze Image (POST Request)

```bash
curl -X POST -F "file=@cotton_image.jpg" http://localhost:5000/api/analyze
```

---

## 📦 Response Format (JSON)

```json
{
  "status": "success",
  "analysis": {
    "stage": "Bursting (Ripped)",
    "stage_confidence": 0.87,
    "health_status": "Pink Bollworm Damage",
    "health_confidence": 0.76,
    "health_score": 68.5,
    "is_ripped": true,
    "has_damage": true
  },
  "recommendations": ["..."]
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
- 🌾 Multi-crop support (🍅 Tomato crop analysis - In Progress)
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

---

# 📌 Contribution Guidelines Reminder

## Before Contributing

- Do not start working on an issue until it is assigned to you.
- Comment on the issue you want to work on.
- Wait for maintainers to assign the issue before creating a PR.

---

## 🔀 Pull Request Guidelines

- Mention the related issue in your PR description using:
  - `Closes #issue_number`
  - `Fixes #issue_number`
- Keep PRs focused on a single issue/topic.
- Add screenshots or demo videos for UI-related changes.
- Sync your branch with the latest repository changes before submitting a PR.

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

## ❤️ Made with Passion by neeru24

⭐ If you found this project helpful, consider giving it a star. ⭐

</div>