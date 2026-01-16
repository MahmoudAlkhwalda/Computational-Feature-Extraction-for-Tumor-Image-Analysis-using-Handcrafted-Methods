from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from skimage import measure
from cancer_features import extract_all_features, read_image, preprocess_image, segment_tumor, get_largest_region
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_visualization(img_rgb, gray, mask, region_mask, region):
    """Create visualization and return as base64 encoded image"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Preprocessed Gray")
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Segmentation Mask")
    axes[2].axis('off')
    
    axes[3].imshow(img_rgb)
    try:
        contours = measure.find_contours(region_mask.astype(float), 0.5)
        for cnt in contours:
            if len(cnt) > 0:
                axes[3].plot(cnt[:, 1], cnt[:, 0], '-r', linewidth=1)
    except Exception:
        pass
    axes[3].set_title("Detected Region Overlay")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, GIF'}), 400
    
    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_rgb = read_image(filepath, as_gray=False)
        gray = preprocess_image(img_rgb)
        mask = segment_tumor(gray)

        df, region, region_mask = extract_all_features(filepath, visualize=False)
        
        
        viz_image = create_visualization(img_rgb, gray, mask, region_mask, region)
        

        features_dict = df.to_dict('records')[0]
        
   
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'features': features_dict,
            'visualization': viz_image,
            'filename': filename
        })
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        data = request.json
        features = data.get('features', {})
        
        df = pd.DataFrame([features])
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='extracted_features.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

