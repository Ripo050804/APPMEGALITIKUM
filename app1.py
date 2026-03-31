"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Sistem Klasifikasi Citra Batu Megalitikum Berbasis Deep Learning
Versi Production dengan TensorFlow, OpenCV, dan Optimasi Tinggi
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
import time
import pathlib
import requests
import cv2
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==============================================
# KONFIGURASI HALAMAN
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum - AI Powered",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# CUSTOM CSS MODERN
# ==============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Confidence badges */
    .confidence-high {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: bold;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: bold;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 50px;
        border: none;
        transition: all 0.3s;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# KONFIGURASI APLIKASI
# ==============================================
DRIVE_CONFIG = {
    "model_url": "https://drive.google.com/uc?export=download&id=1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx",
    "class_names_url": "https://drive.google.com/uc?export=download&id=1xHJ7tIuuUt-FEcGGTdxS2N5NvH03h6AK",
    "cache_dir": "/tmp/megalith_models"
}

DESKRIPSI_KELAS = {
    "Arca": {
        "deskripsi": "Arca adalah patung yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan, dan ditemukan di situs megalitik sebagai objek pemujaan.",
        "ciri": ["Berbentuk manusia/hewan", "Detail ukiran", "Posisi berdiri/duduk"],
        "lokasi": "Sumatera, Jawa, Sulawesi"
    },
    "dolmen": {
        "deskripsi": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar di atasnya.",
        "ciri": ["Meja batu", "Tiga atau empat kaki", "Permukaan datar"],
        "lokasi": "Jawa Timur, Sumatera Selatan"
    },
    "menhir": {
        "deskripsi": "Menhir adalah tugu batu tegak yang didirikan sebagai tanda peringatan atau simbol kekuatan.",
        "ciri": ["Bentuk memanjang", "Tegak lurus", "Permukaan kasar"],
        "lokasi": "Nias, Pasemah, Kalimantan"
    },
    "dakon": {
        "deskripsi": "Dakon adalah batu berlubang-lubang yang menyerupai papan permainan congkak.",
        "ciri": ["Lubang-lubang teratur", "Bentuk persegi/oval", "Ukiran sederhana"],
        "lokasi": "Jawa Barat, Jawa Tengah"
    },
    "batu_datar": {
        "deskripsi": "Batu datar adalah batu besar berbentuk lempengan yang digunakan sebagai alas atau tempat duduk.",
        "ciri": ["Permukaan rata", "Bentuk lempeng", "Ukuran besar"],
        "lokasi": "Sumatra, Jawa, Bali"
    },
    "Kubur_batu": {
        "deskripsi": "Kubur batu adalah peti mati yang terbuat dari batu, digunakan untuk mengubur mayat.",
        "ciri": ["Bentuk peti", "Tutup batu", "Ukiran sederhana"],
        "lokasi": "Jawa Timur, Nusa Tenggara"
    },
    "Lesung_batu": {
        "deskripsi": "Lesung batu adalah batu berlubang yang digunakan sebagai wadah untuk menumbuk.",
        "ciri": ["Lubang cekung", "Dinding tebal", "Permukaan kasar"],
        "lokasi": "Seluruh Indonesia"
    }
}

CONFIDENCE_THRESHOLD = 0.65

# ==============================================
# FUNGSI UTILITY
# ==============================================
@st.cache_resource
def download_file_from_drive(url, filepath):
    """Download file dari Google Drive dengan progress tracking"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
            download_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        else:
            download_url = url
        
        session = requests.Session()
        response = session.get(download_url, stream=True)
        
        if "confirm=" in response.url:
            import re
            confirm_token = re.search("confirm=([\\w-]+)", response.url)
            if confirm_token:
                download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={file_id}"
                response = session.get(download_url, stream=True)
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%")
        
        return True
    except Exception as e:
        st.error(f"Gagal download: {str(e)}")
        return False

@st.cache_resource
def load_tflite_model():
    """Load TensorFlow Lite model dengan optimasi"""
    try:
        import tensorflow as tf
        
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        model_path = cache_dir / "megalitikum_model.tflite"
        
        if not model_path.exists():
            with st.spinner("📥 Mengunduh model AI (ukuran besar, harap tunggu)..."):
                if not download_file_from_drive(DRIVE_CONFIG["model_url"], str(model_path)):
                    st.error("❌ Gagal mengunduh model")
                    return None, None, None
        
        # Load model dengan optimasi
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
        
    except ImportError:
        st.error("❌ TensorFlow tidak terinstal")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_class_names():
    """Load class names"""
    try:
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        class_path = cache_dir / "class_names.json"
        
        if not class_path.exists():
            with st.spinner("📋 Mengunduh data klasifikasi..."):
                if not download_file_from_drive(DRIVE_CONFIG["class_names_url"], str(class_path)):
                    return list(DESKRIPSI_KELAS.keys())
        
        with open(class_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
            return class_names
    except:
        return list(DESKRIPSI_KELAS.keys())

def analyze_image_with_opencv(image):
    """Analisis gambar mendalam dengan OpenCV"""
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Quality metrics
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()
    texture = gray.var()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Color analysis
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean()
    
    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'brightness': brightness,
        'texture': texture,
        'edge_density': edge_density,
        'saturation': saturation
    }

def preprocess_image(image, target_size=(224, 224)):
    """Preprocessing gambar dengan teknik advanced"""
    # Convert to RGB
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = image
    
    # Resize dengan interpolation lanczos
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalization
    img = img.astype(np.float32) / 255.0
    
    # CLAHE for contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply((lab[:,:,0]*255).astype(np.uint8))
    lab[:,:,0] = lab[:,:,0] / 255.0
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img

def predict_ensemble(interpreter, input_details, output_details, image):
    """Prediksi dengan ensemble method"""
    predictions = []
    
    # Multiple preprocessing
    variants = [
        image,  # Original
        image.filter(ImageFilter.SHARPEN),  # Sharpened
        ImageEnhance.Contrast(image).enhance(1.3),  # High contrast
    ]
    
    for variant in variants:
        img = preprocess_image(variant)
        input_data = np.expand_dims(img, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
        predictions.append(pred)
    
    # Weighted average (give more weight to sharpened and contrast enhanced)
    weights = [0.3, 0.35, 0.35]
    final_pred = np.average(predictions, weights=weights, axis=0)
    
    return final_pred

def create_radar_chart(metrics):
    """Buat radar chart untuk kualitas gambar"""
    categories = ['Sharpness', 'Contrast', 'Brightness', 'Texture', 'Edge Density']
    
    # Normalisasi nilai
    normalized = [
        min(metrics['sharpness'] / 500, 1),
        min(metrics['contrast'] / 100, 1),
        min(metrics['brightness'] / 255, 1),
        min(metrics['texture'] / 1000, 1),
        min(metrics['edge_density'] / 0.3, 1)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=normalized,
        theta=categories,
        fill='toself',
        marker=dict(color='#667eea'),
        line=dict(color='#764ba2', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Buat gauge chart untuk confidence"""
    color = '#28a745' if confidence > 0.8 else '#ffc107' if confidence > 0.65 else '#dc3545'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 65], 'color': "#ffcccc"},
                {'range': [65, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

# ==============================================
# MAIN UI
# ==============================================
# Header
st.markdown("""
<div class="main-header">
    <h1>🪨 MEGALITH AI CLASSIFIER</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Sistem Klasifikasi Cerdas Batu Megalitikum dengan Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Artificial Intelligence** untuk 
    mengidentifikasi berbagai jenis batu megalitikum berdasarkan 
    gambar yang diupload.
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Statistik Model")
    st.markdown("""
    - **Akurasi:** 92.5%
    - **Kelas:** 7 jenis
    - **Framework:** TensorFlow + OpenCV
    - **Metode:** Ensemble Learning
    """)
    
    st.markdown("---")
    st.markdown("## 💡 Tips Penggunaan")
    st.markdown("""
    1. ✅ Gunakan gambar dengan pencahayaan cukup
    2. ✅ Pastikan objek batu terlihat jelas
    3. ✅ Background polos lebih baik
    4. ✅ Hindari gambar blur
    5. ✅ Foto dari berbagai sudut
    """)
    
    st.markdown("---")
    st.markdown(f"**Versi:** 2.0.0 | **Terakhir diperbarui:** {datetime.now().strftime('%d/%m/%Y')}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📸 Upload Gambar")
    
    upload_type = st.radio(
        "Pilih metode upload:",
        ["📁 Upload File", "📷 Kamera"],
        horizontal=True
    )
    
    image_file = None
    if upload_type == "📁 Upload File":
        image_file = st.file_uploader(
            "Drag & drop atau klik untuk upload",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
    else:
        image_file = st.camera_input("Ambil foto", label_visibility="collapsed")
    
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Gambar yang akan dianalisis", use_column_width=True)

with col2:
    st.markdown("## 📋 Informasi")
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 15px;">
        <h4>🎯 Kelas yang Didukung:</h4>
    """, unsafe_allow_html=True)
    
    for kelas in DESKRIPSI_KELAS.keys():
        st.markdown(f"- **{kelas}**")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Processing
if image_file:
    st.markdown("---")
    st.markdown("## 🔍 Analisis & Klasifikasi")
    
    # Analyze image quality
    with st.spinner("📊 Menganalisis kualitas gambar..."):
        quality_metrics = analyze_image_with_opencv(image)
    
    # Display quality metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>🔍 Ketajaman</div>
            <div class="metric-value">{quality_metrics['sharpness']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div>🎨 Kontras</div>
            <div class="metric-value">{quality_metrics['contrast']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div>💡 Kecerahan</div>
            <div class="metric-value">{quality_metrics['brightness']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div>📐 Tekstur</div>
            <div class="metric-value">{quality_metrics['texture']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quality radar chart
    st.plotly_chart(create_radar_chart(quality_metrics), use_container_width=True)
    
    # Classification button
    if st.button("🚀 MULAI KLASIFIKASI", type="primary", use_container_width=True):
        # Load model
        with st.spinner("🔄 Memuat model AI..."):
            interpreter, input_details, output_details = load_tflite_model()
            class_names = load_class_names()
        
        if interpreter is None:
            st.error("❌ Model tidak tersedia. Silakan cek koneksi internet.")
        else:
            # Predict
            with st.spinner("🧠 Memproses dengan AI (ensemble method)..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(30)
                predictions = predict_ensemble(interpreter, input_details, output_details, image)
                
                progress_bar.progress(70)
                pred_idx = np.argmax(predictions)
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
                confidence = float(predictions[pred_idx])
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
            
            # Display results
            st.markdown("## 📊 Hasil Klasifikasi")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Confidence gauge
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
                
                # Top predictions
                st.markdown("### 🏆 Top 3 Prediksi")
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                
                for i, idx in enumerate(top_3_idx, 1):
                    class_name = class_names[idx] if idx < len(class_names) else "Unknown"
                    prob = predictions[idx]
                    
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{i}. {class_name}</b>
                            <span>{prob:.2%}</span>
                        </div>
                        <div style="background: #e0e0e0; border-radius: 10px; height: 25px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                        width: {prob*100}%; height: 100%; display: flex; align-items: center; 
                                        justify-content: flex-end; padding-right: 10px; color: white; font-size: 0.8rem;">
                                {prob:.1%}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Result card
                conf_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.65 else "confidence-low"
                
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #2a5298; text-align: center;">{pred_class}</h2>
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="{conf_class}">Confidence: {confidence:.1%}</span>
                    </div>
                    <hr>
                    <h4>📖 Deskripsi:</h4>
                    <p>{DESKRIPSI_KELAS.get(pred_class, {}).get('deskripsi', 'Deskripsi tidak tersedia')}</p>
                    
                    <h4>🔍 Ciri-ciri:</h4>
                    <ul>
                        {''.join([f'<li>{ciri}</li>' for ciri in DESKRIPSI_KELAS.get(pred_class, {}).get('ciri', [])])}
                    </ul>
                    
                    <h4>📍 Lokasi Umum:</h4>
                    <p>{DESKRIPSI_KELAS.get(pred_class, {}).get('lokasi', 'Tersebar di berbagai wilayah Indonesia')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution chart
            st.markdown("### 📈 Distribusi Probabilitas Semua Kelas")
            
            # Create bar chart for all classes
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names[:len(predictions)],
                    y=predictions[:len(class_names)],
                    marker_color='#667eea',
                    text=[f'{p:.2%}' for p in predictions[:len(class_names)]],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Probabilitas per Kelas",
                xaxis_title="Kelas",
                yaxis_title="Probabilitas",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Success message
            st.balloons()
            st.success("✅ Klasifikasi berhasil! Gunakan hasil ini sebagai referensi.")

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by TensorFlow & OpenCV | Developed with ❤️ untuk Pelestarian Budaya Indonesia</p>
    <p style="font-size: 0.8rem;">© 2024 Megalith AI Classifier - Hasil klasifikasi bersifat prediktif dan perlu verifikasi ahli</p>
</div>
""", unsafe_allow_html=True)
