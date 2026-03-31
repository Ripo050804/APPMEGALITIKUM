"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Sistem Klasifikasi Citra Batu Megalitikum Berbasis Deep Learning
Versi Production - Siap Deployment di Streamlit Cloud
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import time
import tempfile
from datetime import datetime
import gdown
import requests

# ==============================================
# KONFIGURASI HALAMAN (HARUS PERTAMA)
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# INISIALISASI SESSION STATE
# ==============================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'interpreter' not in st.session_state:
    st.session_state.interpreter = None

# ==============================================
# CUSTOM CSS
# ==============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .main-header p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
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
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 50px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# KONSTANTA GLOBAL
# ==============================================

# Google Drive File IDs (ganti dengan ID file Anda)
MODEL_FILE_ID = "1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx"
CLASS_NAMES_FILE_ID = "1xHJ7tIuuUt-FEcGGTdxS2N5NvH03h6AK"

# Gunakan temporary directory untuk cache (kompatibel dengan semua OS)
CACHE_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(CACHE_DIR, "megalitikum_model.tflite")
CLASS_NAMES_PATH = os.path.join(CACHE_DIR, "class_names.json")

# Threshold konfidensi
CONFIDENCE_THRESHOLD = 0.65

# Target size untuk model
TARGET_SIZE = (224, 224)

# Deskripsi kelas (fallback jika file class_names.json tidak tersedia)
DESKRIPSI_KELAS_FALLBACK = {
    "Arca": {
        "deskripsi": "Arca adalah patung yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan.",
        "ciri": ["Berbentuk manusia/hewan", "Detail ukiran", "Posisi berdiri/duduk"],
        "lokasi": "Sumatera, Jawa, Sulawesi"
    },
    "dolmen": {
        "deskripsi": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar.",
        "ciri": ["Meja batu", "Tiga atau empat kaki", "Permukaan datar"],
        "lokasi": "Jawa Timur, Sumatera Selatan"
    },
    "menhir": {
        "deskripsi": "Menhir adalah tugu batu tegak sebagai tanda peringatan atau simbol kekuatan.",
        "ciri": ["Bentuk memanjang", "Tegak lurus", "Permukaan kasar"],
        "lokasi": "Nias, Pasemah, Kalimantan"
    },
    "dakon": {
        "deskripsi": "Dakon adalah batu berlubang-lubang seperti papan permainan congkak.",
        "ciri": ["Lubang-lubang teratur", "Bentuk persegi/oval", "Ukiran sederhana"],
        "lokasi": "Jawa Barat, Jawa Tengah"
    },
    "batu_datar": {
        "deskripsi": "Batu datar adalah batu besar berbentuk lempengan untuk alas atau tempat duduk.",
        "ciri": ["Permukaan rata", "Bentuk lempeng", "Ukuran besar"],
        "lokasi": "Sumatra, Jawa, Bali"
    },
    "Kubur_batu": {
        "deskripsi": "Kubur batu adalah peti mati dari batu untuk mengubur mayat.",
        "ciri": ["Bentuk peti", "Tutup batu", "Ukiran sederhana"],
        "lokasi": "Jawa Timur, Nusa Tenggara"
    },
    "Lesung_batu": {
        "deskripsi": "Lesung batu adalah batu berlubang untuk menumbuk bahan makanan.",
        "ciri": ["Lubang cekung", "Dinding tebal", "Permukaan kasar"],
        "lokasi": "Seluruh Indonesia"
    }
}

# ==============================================
# FUNGSI DOWNLOAD MODEL
# ==============================================
@st.cache_resource
def download_model():
    """Download model dan class names dari Google Drive menggunakan gdown"""
    try:
        # Download model
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Mengunduh model AI (ukuran besar, harap tunggu)..."):
                url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("✅ Model berhasil diunduh!")
        
        # Download class names
        if not os.path.exists(CLASS_NAMES_PATH):
            with st.spinner("📋 Mengunduh data klasifikasi..."):
                url = f"https://drive.google.com/uc?id={CLASS_NAMES_FILE_ID}"
                gdown.download(url, CLASS_NAMES_PATH, quiet=False)
                st.success("✅ Data kelas berhasil diunduh!")
        
        return True
    except Exception as e:
        st.error(f"❌ Gagal mengunduh file: {str(e)}")
        return False

# ==============================================
# LOAD MODEL TENSORFLOW LITE
# ==============================================
@st.cache_resource
def load_tflite_model():
    """Load TensorFlow Lite model"""
    try:
        import tensorflow as tf
        
        # Download model terlebih dahulu
        if not download_model():
            return None, None, None
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
        
    except ImportError:
        st.warning("⚠️ TensorFlow tidak terinstal. Aplikasi berjalan dalam mode demo.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# ==============================================
# LOAD CLASS NAMES
# ==============================================
@st.cache_data
def load_class_names():
    """Load class names dari file JSON"""
    try:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
                return class_names
        else:
            return list(DESKRIPSI_KELAS_FALLBACK.keys())
    except Exception as e:
        st.warning(f"Gagal load class names: {str(e)}")
        return list(DESKRIPSI_KELAS_FALLBACK.keys())

# ==============================================
# FUNGSI PREPROCESSING GAMBAR (Tanpa OpenCV)
# ==============================================
def preprocess_image(image):
    """
    Preprocessing gambar untuk model
    Menggunakan PIL saja untuk menghindari dependensi OpenCV yang berat
    """
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize dengan kualitas tinggi
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # Konversi ke numpy array dan normalisasi
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Enhance contrast sederhana
        img_array = np.power(img_array, 0.8)  # Gamma correction sederhana
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing: {str(e)}")
        return None

# ==============================================
# FUNGSI ANALISIS KUALITAS GAMBAR (Tanpa OpenCV)
# ==============================================
def analyze_image_quality(image):
    """Analisis kualitas gambar menggunakan PIL dan NumPy"""
    try:
        # Convert to grayscale
        gray = image.convert('L')
        gray_array = np.array(gray, dtype=np.float32)
        
        # Hitung metrik
        brightness = np.mean(gray_array)
        contrast = np.std(gray_array)
        texture = np.var(gray_array)
        
        # Estimasi ketajaman (gradient sederhana)
        if gray_array.shape[0] > 2 and gray_array.shape[1] > 2:
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            sharpness = np.mean(grad_x) + np.mean(grad_y)
        else:
            sharpness = 100
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'texture': texture,
            'sharpness': sharpness
        }
    except Exception as e:
        return {
            'brightness': 128,
            'contrast': 64,
            'texture': 500,
            'sharpness': 200
        }

# ==============================================
# FUNGSI PREDIKSI
# ==============================================
def predict_image(interpreter, input_details, output_details, image):
    """Prediksi gambar menggunakan model TFLite"""
    try:
        # Preprocess
        img_array = preprocess_image(image)
        if img_array is None:
            return None
        
        # Prepare input
        input_data = np.expand_dims(img_array, axis=0)
        
        # Predict
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        return predictions
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return None

# ==============================================
# FUNGSI SIMULASI PREDIKSI (Mode Demo)
# ==============================================
def simulate_prediction(image, class_names):
    """Simulasi prediksi untuk mode demo"""
    try:
        # Analisis sederhana untuk simulasi
        quality = analyze_image_quality(image)
        
        # Buat prediksi dummy berdasarkan tekstur
        np.random.seed(int(quality['texture']) % 10000)
        predictions = np.random.rand(len(class_names))
        predictions = predictions / np.sum(predictions)
        
        return predictions
    except Exception:
        # Fallback: distribusi uniform
        predictions = np.ones(len(class_names)) / len(class_names)
        return predictions

# ==============================================
# FUNGSI GET DESKRIPSI
# ==============================================
def get_description(class_name):
    """Mendapatkan deskripsi kelas"""
    if class_name in DESKRIPSI_KELAS_FALLBACK:
        return DESKRIPSI_KELAS_FALLBACK[class_name]
    return {
        "deskripsi": f"Deskripsi untuk {class_name} tidak tersedia.",
        "ciri": ["Belum terdefinisi"],
        "lokasi": "Belum terdefinisi"
    }

# ==============================================
# HEADER UTAMA
# ==============================================
st.markdown("""
<div class="main-header">
    <h1>🪨 KLASIFIKASI BATU MEGALITIKUM</h1>
    <p>Sistem Identifikasi Cerdas Berbasis Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ==============================================
# SIDEBAR
# ==============================================
with st.sidebar:
    st.markdown("## 🎯 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Deep Learning** untuk mengidentifikasi 
    berbagai jenis batu megalitikum dari gambar yang diupload.
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Informasi")
    st.markdown("""
    - **Kelas:** 7 jenis
    - **Framework:** TensorFlow Lite
    - **Akurasi:** 90%+
    """)
    
    st.markdown("---")
    st.markdown("## 💡 Tips Penggunaan")
    st.markdown("""
    1. ✅ Pencahayaan cukup
    2. ✅ Objek batu jelas
    3. ✅ Background polos
    4. ✅ Hindari gambar blur
    """)
    
    st.markdown("---")
    st.markdown("## 📞 Kontak")
    st.markdown("Untuk keperluan akademik dan penelitian.")

# ==============================================
# LOAD MODEL
# ==============================================
with st.spinner("🔄 Memuat model AI..."):
    interpreter, input_details, output_details = load_tflite_model()
    class_names = load_class_names()
    
    # Cek apakah model berhasil dimuat
    use_real_model = interpreter is not None
    
    if not use_real_model:
        st.info("ℹ️ Mode Demo: Aplikasi berjalan dengan simulasi. Install TensorFlow untuk hasil akurat.")

# ==============================================
# MAIN CONTENT - UPLOAD AREA
# ==============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📸 Upload Gambar")
    
    # Pilihan upload
    upload_type = st.radio(
    "Pilih sumber gambar:",
    ["📁 Upload File", "📷 Kamera"],
    horizontal=True
    )
    
    image_file = None
    if upload_type == "📁 Upload File":
        image_file = st.file_uploader(
            "Pilih file gambar (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
    else:
        image_file = st.camera_input("Ambil foto", label_visibility="collapsed")
    
    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {str(e)}")
            image = None

with col2:
    st.markdown("## 🎯 Kelas yang Didukung")
    st.markdown('<div style="background: white; padding: 1rem; border-radius: 15px;">', unsafe_allow_html=True)
    for kelas in class_names:
        st.markdown(f"- **{kelas}**")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
= PROSES KLASIFIKASI
# ==============================================
if image_file and image is not None:
    st.markdown("---")
    st.markdown("## 🔍 Analisis & Klasifikasi")
    
    # Analisis kualitas gambar
    with st.spinner("📊 Menganalisis kualitas gambar..."):
        quality_metrics = analyze_image_quality(image)
    
    # Tampilkan metrik kualitas
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
    
    # Peringatan kualitas
    if quality_metrics['brightness'] < 50:
        st.warning("⚠️ Gambar terlalu gelap. Hasil klasifikasi mungkin kurang akurat.")
    elif quality_metrics['brightness'] > 200:
        st.warning("⚠️ Gambar terlalu terang. Hasil klasifikasi mungkin kurang akurat.")
    
    if quality_metrics['sharpness'] < 100:
        st.warning("⚠️ Gambar kurang tajam. Hasil klasifikasi mungkin kurang akurat.")
    
    # Tombol klasifikasi
    if st.button("🚀 MULAI KLASIFIKASI", type="primary", use_container_width=True):
        with st.spinner("🧠 Memproses dengan AI..."):
            progress_bar = st.progress(0)
            
            progress_bar.progress(30)
            
            # Prediksi
            if use_real_model:
                predictions = predict_image(interpreter, input_details, output_details, image)
            else:
                predictions = simulate_prediction(image, class_names)
            
            progress_bar.progress(70)
            
            if predictions is not None:
                # Ambil hasil terbaik
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
                confidence = float(predictions[pred_idx])
                
                # Top 3 prediksi
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = [(class_names[i], float(predictions[i])) for i in top_3_idx if i < len(class_names)]
                
                progress_bar.progress(100)
                time.sleep(0.3)
                progress_bar.empty()
                
                # Tampilkan hasil
                st.markdown("## 📊 Hasil Klasifikasi")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Confidence display
                    if confidence >= 0.8:
                        conf_class = "confidence-high"
                    elif confidence >= 0.65:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="{conf_class}">Confidence: {confidence:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top predictions
                    st.markdown("### 🏆 Top 3 Prediksi")
                    
                    for i, (kelas, prob) in enumerate(top_3, 1):
                        bar_width = prob * 100
                        st.markdown(f"""
                        <div style="margin: 15px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <b>{i}. {kelas}</b>
                                <span>{prob:.2%}</span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                            width: {bar_width}%; height: 100%; display: flex; align-items: center; 
                                            justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold;">
                                    {prob:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Result card
                    info = get_description(pred_class)
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style="color: #2a5298; text-align: center; margin-bottom: 1rem;">{pred_class}</h2>
                        <hr>
                        <h4>📖 Deskripsi:</h4>
                        <p style="text-align: justify;">{info.get('deskripsi', 'Deskripsi tidak tersedia')}</p>
                        
                        <h4>🔍 Ciri-ciri:</h4>
                        <ul>
                            {''.join([f'<li>{ciri}</li>' for ciri in info.get('ciri', [])])}
                        </ul>
                        
                        <h4>📍 Lokasi Penemuan:</h4>
                        <p>{info.get('lokasi', 'Tersebar di berbagai wilayah Indonesia')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mode demo warning
                if not use_real_model:
                    st.info("ℹ️ **Mode Demo**: Hasil ini adalah simulasi. Install TensorFlow untuk hasil akurat.")
                
                # Success message
                st.balloons()
                st.success("✅ Klasifikasi selesai!")
                
            else:
                progress_bar.empty()
                st.error("❌ Gagal melakukan klasifikasi. Silakan coba lagi.")

# ==============================================
# FOOTER
# ==============================================
st.markdown("""
<div class="footer">
    <p>Powered by TensorFlow Lite | Aplikasi Klasifikasi Batu Megalitikum</p>
    <p style="font-size: 0.8rem;">© 2024 - Untuk Keperluan Penelitian dan Skripsi</p>
</div>
""", unsafe_allow_html=True)
