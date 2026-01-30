from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='template')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image to match model input"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 100, 100, 3)
    return img_array

def predict_image(image_path):
    """Make prediction on the image"""
    try:
        img_array = preprocess_image(image_path)
        print(f"Image array shape: {img_array.shape}")
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        print(f"Raw prediction value: {prediction}")
        
        # Threshold-based classification (tuned for your model)
        # Based on test results: Dog ~0.02-0.22, Neither ~0.45-0.62, Cat ~0.88
        if prediction < 0.35:
            label = "Dog"
            confidence = (1 - prediction) * 100
        elif prediction > 0.65:
            label = "Cat"
            confidence = prediction * 100
        else:
            label = "Neither"
            confidence = (1 - abs(prediction - 0.5) * 2) * 100
        
        print(f"Label: {label}, Confidence: {confidence}")
        return label, float(confidence)
    
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        label, confidence = predict_image(filepath)
        
        if label is None or confidence is None:
            return jsonify({'error': 'Failed to process image. Check Flask console for details.'}), 500
        
        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2),
            'success': True
        }), 200
    
    except Exception as e:
        print(f"Error in /predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
    