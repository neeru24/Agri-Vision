# 🌱 Agri-Vision: Cotton Crop Analysis System

Agri-Vision is an AI-powered system that analyzes cotton crop images to determine growth stages and health conditions.  
It helps farmers and researchers make informed decisions about crop management and harvest timing.

---

## 📌 Overview

Agri-Vision uses deep learning and computer vision techniques to:

- Detect cotton growth phases  
- Identify crop health issues and diseases  
- Provide confidence scores and actionable recommendations  
- Offer both a web interface and a REST API  

---

## ✨ Features

- 🌿 **Growth Phase Detection** (4 stages)  
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

## 📊 Growth Phases Detected

- 🌱 **Vegetative / Budding** – Early growth stage  
- 🌸 **Flowering** – Flower development stage  
- 🍂 **Bursting (Ripped)** – Cotton bolls opening  
- ✅ **Harvest Ready** – Optimal harvest time  

---

## 🩺 Health Issues Identified

- ✅ **Healthy** – No issues detected  
- 🐛 **Pink Bollworm Damage** – Pest infestation detected  
- 🎨 **Discoloration** – Nutrient or water deficiency  
- ⚠️ **Other Damage** – Miscellaneous crop issues  

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

# 📈 Model Performance

- ✅ Accuracy: **~85–90%** on test data  
- ⚡ Processing Time: **< 2 seconds per image**  
- 🖼️ Supported Formats: **JPG, PNG, JPEG**  
- 📦 Maximum File Size: **10MB**

---

# 🚀 Future Enhancements

- 📱 Mobile application support  
- 🎥 Real-time video analysis  
- 🌾 Multi-crop support  
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

---

<div align="center">

## ❤️ Made with Passion by neeru24

⭐ If you found this project helpful, consider giving it a star. ⭐

</div>