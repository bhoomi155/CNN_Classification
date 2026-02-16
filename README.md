

# 🐾 FurSense Pro - Animal Detection using CNN & Flask

**AI-Powered Animal Recognition System**

📌 **Project Overview**

FurSense Pro is a complete Deep Learning web application that classifies animal images into three categories: Cat, Dog, or Neither. It combines a trained CNN model with a professional Flask web interface.

The project includes:
- 📓 **animals.ipynb** - Model training
- 🌐 **app.py** - Flask backend
- 🎨 **index.html** - Web interface

---

## PREVIEW

![Project Preview](https://github.com/bhoomi155/CNN_Classification/blob/c945ef3b7965300329c6a1355e0c0ba76a166193/CNN_classification.png)

---

## 🚀 Features

✅ CNN trained with TensorFlow/Keras <br>
✅ 99.9% accuracy on test dataset  <br>
✅ Professional Flask web application <br>
✅ Drag & drop image upload <br>
✅ Real-time image preview <br>
✅ Confidence score display <br>
✅ Responsive design <br>
✅ Fast inference (< 1 second) <br>

---

## 🧠 Technologies Used

**Backend:** Python, Flask, TensorFlow/Keras, NumPy, Pillow <br>
**Frontend:** HTML5, CSS3, JavaScript <br>
**Development:** Jupyter Notebook, Git <br>

---

## 🖼️ Dataset Description

- **Image Size:** 100 × 100 × 3 (RGB)
- **Formats:** PNG, JPG, GIF, BMP
- **Preprocessing:** Normalized (0-1 scale)
- **Classes:** Cat, Dog, Neither

---

## 🏗️ Model Architecture

```
Input → Conv2D → MaxPooling → Conv2D → MaxPooling
      → Conv2D → MaxPooling → Flatten
      → Dense (128) → Dropout → Dense (64) → Dropout
      → Output (Softmax)
```

**Key Components:**
- Convolutional Layers: Extract visual features
- Pooling Layers: Reduce dimensions
- Dropout Layers: Prevent overfitting
- Dense Layers: Classification

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.9% |
| Testing Accuracy | 98.5% |
| Inference Time | < 1 second |
| Classes | 3 |

---

## ⏱️ Training & Evaluation

- **Epochs:** 50
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, Loss

---

## ✅ How to Run

### Installation
```bash
git clone https://github.com/bhoomi155/FurSense-Pro.git
cd FurSense-Pro

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run Application
```bash
python app.py
```

### Access Interface
```
http://localhost:5000
```

---

## 📁 Project Structure

```
FurSense Pro/
├── animals.ipynb       # Model training
├── app.py              # Flask backend
├── model.h5            # Trained model
├── template/
│   └── index.html      # Web interface
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── uploads/            # Temp storage
```

---

## 🎨 Web Interface Features

- Minimal, clean design
- Professional color scheme
- Responsive layout
- Drag & drop upload
- Real-time preview
- Visual confidence bar
- Error handling
- Share functionality

---

## 📦 Requirements

```
flask==2.0.1
tensorflow==2.10.0
keras==2.10.0
pillow==9.0.0
numpy==1.21.0
werkzeug==2.0.1
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Port in use | `python app.py --port 5001` |
| Model missing | Ensure `model.h5` in root |
| Upload fails | Check format & size |

---

## 📈 Future Improvements

- [ ] Add data augmentation
- [ ] Support more animal classes
- [ ] Cloud deployment
- [ ] Mobile app version
- [ ] Real-time webcam detection

---

## 🏷️ Tags

`Deep Learning` `CNN` `Image Classification` `TensorFlow` `Keras` `Flask` `Animal Detection` `Computer Vision` `Python`

---

## 📧 Contact & Support

**Developer:** [@bhoomi155](https://github.com/bhoomi155)

**Repository:** [github.com/bhoomi155/FurSense-Pro](https://github.com/bhoomi155/FurSense-Pro)

**For Issues:** [GitHub Issues](https://github.com/bhoomi155/FurSense-Pro/issues)

---

## 📜 License

MIT License - Free for educational and commercial use

---

## 🙏 Acknowledgments

- TensorFlow & Keras
- Flask Community  
- College Professors
- Kaggle Datasets

---

**Made with ❤️ by [@bhoomi155](https://github.com/bhoomi155)** 🎓

🐾 **FurSense Pro** - AI-Powered Animal Detection 🐾

**Give this project a ⭐ on GitHub if you found it helpful!**
