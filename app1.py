"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Sistem Klasifikasi Citra Batu Megalitikum Berbasis Deep Learning
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
import time
import pathlib
import requests
from io import BytesIO
import base64
import sys

# ==============================================
# KONFIGURASI HALAMAN
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🪨",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ==============================================
# CUSTOM CSS UNTUK MOBILE & TAMPILAN PROFESIONAL
# ==============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .main-header p {
        margin: 0.5rem 0 0;
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }
    
    .success-card {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .warning-card {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .error-card {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .result-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem; }
        .main-header p { font-size: 0.85rem; }
        .info-card, .result-box { padding: 1rem; }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .st-emotion-cache-16txtl3 { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================
# KONFIGURASI GOOGLE DRIVE - FILE ID DARI LINK ANDA
# ==============================================
DRIVE_CONFIG = {
    # Model: https://drive.google.com/file/d/1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx/view
    "model_url": "https://drive.google.com/uc?export=download&id=1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx",
    
    # Class names: https://drive.google.com/file/d/1xHJ7tIuuUt-FEcGGTdxS2N5NvH03h6AK/view
    "class_names_url": "https://drive.google.com/uc?export=download&id=1xHJ7tIuuUt-FEcGGTdxS2N5NvH03h6AK",
    
    "cache_dir": "/tmp/megalith_models"
}

# ==============================================
# DESKRIPSI KELAS
# ==============================================
DESKRIPSI_KELAS = {
    "Arca": "Arca adalah patung yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan, dan ditemukan di situs megalitik sebagai objek pemujaan.",
    "dolmen": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar di atasnya. Digunakan sebagai tempat meletakkan sesaji atau untuk upacara.",
    "menhir": "Menhir adalah tugu batu tegak yang didirikan sebagai tanda peringatan atau simbol kekuatan. Biasanya ditemukan berdiri sendiri atau berkelompok.",
    "dakon": "Dakon adalah batu berlubang-lubang yang menyerupai papan permainan congkak. Diduga digunakan untuk ritual atau permainan tradisional.",
    "batu_datar": "Batu datar adalah batu besar berbentuk lempengan yang mungkin digunakan sebagai alas atau tempat duduk dalam upacara adat.",
    "Kubur_batu": "Kubur batu adalah peti mati yang terbuat dari batu, digunakan untuk mengubur mayat pada masa megalitik. Biasanya ditemukan di dalam tanah.",
    "Lesung_batu": "Lesung batu adalah batu berlubang yang digunakan sebagai wadah untuk menumbuk atau menghaluskan bahan makanan pada masa megalitikum."
}

# ==============================================
# KONFIGURASI
# ==============================================
CONFIDENCE_THRESHOLD = 0.60

# ==============================================
# FUNGSI DOWNLOAD FILE DARI GOOGLE DRIVE
# ==============================================
def download_file_from_drive(url, filepath):
    """Download file dari Google Drive dengan handling redirect"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Extract file ID
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
            download_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        else:
            download_url = url
        
        # Session untuk handle cookies
        session = requests.Session()
        
        # Request pertama untuk mendapatkan confirmation token jika diperlukan
        response = session.get(download_url, stream=True)
        
        # Cek jika perlu confirmation
        if "confirm=" in response.url or response.status_code == 403:
            # Extract confirmation token
            import re
            confirm_token = re.search("confirm=([\\w-]+)", response.url)
            if confirm_token:
                download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={file_id}"
                response = session.get(download_url, stream=True)
        
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
        
    except Exception as e:
        st.error(f"Gagal download file: {str(e)}")
        return False

# ==============================================
# FUNGSI ANALISIS GAMBAR SEDERHANA
# ==============================================
def analyze_image_brightness(image):
    """Analisis kecerahan gambar"""
    gray = image.convert('L')
    histogram = gray.histogram()
    pixels = sum(histogram)
    brightness = sum(i * w for i, w in enumerate(histogram)) / pixels
    return brightness

def analyze_image_texture(image):
    """Analisis tekstur gambar"""
    img_array = np.array(image.convert('L'))
    if img_array.size == 0:
        return 500
    texture_variance = np.var(img_array)
    return texture_variance

def analyze_color_dominance(image):
    """Analisis dominasi warna"""
    img_array = np.array(image.convert('RGB'))
    if img_array.size == 0:
        return 0, 0, 0
    
    r_mean = np.mean(img_array[:,:,0])
    g_mean = np.mean(img_array[:,:,1])
    b_mean = np.mean(img_array[:,:,2])
    
    return r_mean, g_mean, b_mean

def is_megalith_image(image):
    """Filter sederhana untuk gambar batu"""
    try:
        # Analisis warna
        r_mean, g_mean, b_mean = analyze_color_dominance(image)
        
        # Deteksi warna hijau (tumbuhan)
        if g_mean > r_mean * 1.3 and g_mean > b_mean * 1.3:
            return False, "Gambar didominasi warna hijau (mungkin tumbuhan)", 0.1
        
        # Deteksi warna biru (langit/air)
        if b_mean > r_mean * 1.4 and b_mean > g_mean * 1.4:
            return False, "Gambar didominasi warna biru (mungkin langit/air)", 0.1
        
        # Analisis tekstur
        texture_variance = analyze_image_texture(image)
        
        if texture_variance < 100:
            return False, "Tekstur terlalu halus untuk dikategorikan batu", 0.3
        
        # Analisis brightness
        brightness = analyze_image_brightness(image)
        
        if brightness < 40:
            return False, "Gambar terlalu gelap", 0.2
        if brightness > 220:
            return False, "Gambar terlalu terang (overexposed)", 0.2
        
        return True, "Gambar memenuhi kriteria analisis", 0.7
        
    except Exception as e:
        return False, f"Error saat analisis: {str(e)}", 0

# ==============================================
# FUNGSI MEMBUAT HTML UNTUK LAPORAN
# ==============================================
def buat_html_hasil(nama_file, kelas, confidence, top3, deskripsi):
    """Buat HTML untuk laporan"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Hasil Klasifikasi Batu Megalitikum</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2a5298;
                text-align: center;
                border-bottom: 3px solid #2a5298;
                padding-bottom: 10px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .info {{
                background: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .result {{
                font-size: 24px;
                font-weight: bold;
                color: #28a745;
                text-align: center;
                margin: 20px 0;
            }}
            .confidence {{
                font-size: 18px;
                text-align: center;
                color: #666;
            }}
            .top3 {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .description {{
                margin: 20px 0;
                line-height: 1.6;
                text-align: justify;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Laporan Klasifikasi Batu Megalitikum</h1>
                <p>Dihasilkan pada: {time.strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
            
            <div class="info">
                <strong>Nama File:</strong> {nama_file}<br>
                <strong>Tanggal Analisis:</strong> {time.strftime('%d/%m/%Y')}
            </div>
            
            <div class="result">
                {kelas}
            </div>
            
            <div class="confidence">
                Tingkat Keyakinan: {confidence:.1%}
            </div>
            
            <div class="top3">
                <h3>Top 3 Prediksi:</h3>
                <ol>
                    {''.join([f'<li><strong>{k}</strong>: {c:.1%}</li>' for k, c in top3])}
                </ol>
            </div>
            
            <div class="description">
                <h3>Deskripsi:</h3>
                <p>{deskripsi}</p>
            </div>
            
            <div class="footer">
                <p>Aplikasi Klasifikasi Batu Megalitikum<br>Berdasarkan Penelitian Deep Learning</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# ==============================================
# SIMULASI MODEL (UNTUK DEMO)
# ==============================================
def simulate_prediction(image):
    """Simulasi prediksi untuk demo jika model tidak tersedia"""
    np.random.seed(hash(str(time.time())) % 2**32)
    
    # Simulasi prediksi berdasarkan analisis sederhana
    img_array = np.array(image.convert('RGB'))
    
    # Analisis sederhana untuk menentukan kelas
    r_mean, g_mean, b_mean = analyze_color_dominance(image)
    texture = analyze_image_texture(image)
    
    # Logika sederhana untuk simulasi
    predictions = np.zeros(len(DESKRIPSI_KELAS))
    
    if texture > 500:
        # Tekstur kasar -> menhir atau arca
        if r_mean > g_mean and r_mean > b_mean:
            predictions[list(DESKRIPSI_KELAS.keys()).index("menhir") if "menhir" in DESKRIPSI_KELAS else 0] = 0.7
            predictions[list(DESKRIPSI_KELAS.keys()).index("Arca") if "Arca" in DESKRIPSI_KELAS else 0] = 0.3
        else:
            predictions[list(DESKRIPSI_KELAS.keys()).index("Arca") if "Arca" in DESKRIPSI_KELAS else 0] = 0.6
            predictions[list(DESKRIPSI_KELAS.keys()).index("menhir") if "menhir" in DESKRIPSI_KELAS else 0] = 0.4
    else:
        # Tekstur halus -> dolmen atau batu datar
        predictions[list(DESKRIPSI_KELAS.keys()).index("dolmen") if "dolmen" in DESKRIPSI_KELAS else 0] = 0.5
        predictions[list(DESKRIPSI_KELAS.keys()).index("batu_datar") if "batu_datar" in DESKRIPSI_KELAS else 0] = 0.5
    
    # Normalisasi
    predictions = predictions / np.sum(predictions)
    
    return predictions

# ==============================================
# LOAD MODEL (DENGAN FALLBACK)
# ==============================================
@st.cache_resource
def load_model():
    """Load model dengan fallback ke simulasi jika TensorFlow tidak tersedia"""
    tensorflow_available = False
    
    # Cek ketersediaan TensorFlow
    try:
        import tensorflow as tf
        tensorflow_available = True
    except ImportError:
        st.warning("⚠️ TensorFlow tidak terinstal. Aplikasi akan berjalan dalam mode DEMO dengan prediksi simulasi.")
        return None, None, None, False
    
    # Jika TensorFlow tersedia, coba load model
    if tensorflow_available:
        try:
            import tensorflow as tf
            
            cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
            model_path = cache_dir / "megalitikum_model.tflite"
            
            # Download model jika belum ada
            if not model_path.exists():
                with st.status("Mengunduh model dari Google Drive...", expanded=True) as status:
                    if not download_file_from_drive(DRIVE_CONFIG["model_url"], str(model_path)):
                        status.update(label="Gagal mengunduh model, beralih ke mode DEMO", state="error")
                        return None, None, None, False
                    status.update(label="Model berhasil diunduh", state="complete")
            
            # Load model
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            return interpreter, input_details, output_details, True
            
        except Exception as e:
            st.error(f"Gagal memuat model: {str(e)}")
            st.warning("Beralih ke mode DEMO dengan prediksi simulasi")
            return None, None, None, False
    
    return None, None, None, False

@st.cache_data
def load_class_names():
    """Load class names dengan fallback ke default"""
    try:
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        class_path = cache_dir / "class_names.json"
        
        # Download jika belum ada
        if not class_path.exists():
            with st.status("Mengunduh data kelas...", expanded=True) as status:
                if not download_file_from_drive(DRIVE_CONFIG["class_names_url"], str(class_path)):
                    status.update(label="Gagal mengunduh data kelas, menggunakan default", state="warning")
                    return list(DESKRIPSI_KELAS.keys())
                status.update(label="Data kelas berhasil diunduh", state="complete")
        
        # Load JSON dengan error handling
        try:
            with open(class_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
                return class_names
        except UnicodeDecodeError:
            # Coba dengan encoding lain
            with open(class_path, 'r', encoding='latin-1') as f:
                class_names = json.load(f)
                return class_names
            
    except Exception as e:
        st.warning(f"Gagal load class names: {str(e)}. Menggunakan default.")
        return list(DESKRIPSI_KELAS.keys())

# ==============================================
# FUNGSI PREDIKSI
# ==============================================
def predict_image(interpreter, input_details, output_details, image, use_real_model):
    """Prediksi gambar (real atau simulasi)"""
    if use_real_model and interpreter is not None:
        # Prediksi dengan model real
        img = image.convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data[0]
    else:
        # Prediksi simulasi untuk demo
        return simulate_prediction(image)

# ==============================================
# UI HEADER
# ==============================================
st.markdown("""
<div class="main-header">
    <h1>Klasifikasi Batu Megalitikum</h1>
    <p>Sistem Identifikasi Otomatis Berbasis Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ==============================================
# SIDEBAR INFO
# ==============================================
with st.sidebar:
    st.markdown("### Panduan Penggunaan")
    st.markdown("""
    1. Pilih sumber gambar (upload atau kamera)
    2. Pastikan foto batu terlihat jelas
    3. Klik tombol klasifikasi
    4. Lihat hasil dan download laporan
    
    **Kelas yang Didukung:**
    - Arca
    - Dolmen
    - Menhir
    - Dakon
    - Batu Datar
    - Kubur Batu
    - Lesung Batu
    """)
    
    st.markdown("---")
    st.markdown("**Kriteria Gambar:**")
    st.markdown("""
    - Format: JPG, JPEG, PNG
    - Pencahayaan cukup
    - Objek batu terlihat jelas
    - Hindari background ramai
    """)
    
    st.markdown("---")
    st.markdown("**Status:**")
    st.markdown("🟢 Aplikasi siap digunakan")

# ==============================================
# LOAD MODEL
# ==============================================
with st.spinner("Memuat model..."):
    interpreter, input_details, output_details, use_real_model = load_model()
    class_names = load_class_names()

# Tampilkan mode aplikasi
if not use_real_model:
    st.info("📢 **Mode Demo**: Aplikasi berjalan dalam mode demonstrasi dengan prediksi simulasi. Untuk hasil akurat, instal TensorFlow dan sediakan model yang valid.")

# ==============================================
# MAIN INTERFACE
# ==============================================
st.markdown("### Upload atau Ambil Foto")

# Pilihan sumber gambar
sumber = st.radio(
    "Pilih sumber gambar:",
    ["Upload File", "Ambil Foto Kamera"],
    horizontal=True,
    label_visibility="collapsed"
)

gambar = None
if sumber == "Upload File":
    gambar = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
        help="Format yang didukung: JPG, JPEG, PNG"
    )
else:
    st.info("Pastikan pencahayaan cukup saat mengambil foto")
    gambar = st.camera_input("Ambil foto")

if gambar:
    # Proses gambar
    image = Image.open(gambar)
    
    # Tampilkan gambar
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Detail Gambar")
        st.write(f"**Ukuran:** {image.size[0]} x {image.size[1]} px")
        st.write(f"**Format:** {image.format if image.format else 'Unknown'}")
        st.write(f"**Mode:** {image.mode}")
    
    # Analisis awal
    st.markdown("### Verifikasi Gambar")
    
    with st.spinner("Memeriksa kualitas gambar..."):
        is_valid, reason, score = is_megalith_image(image)
    
    if not is_valid:
        st.markdown(f'<div class="error-card"><strong>Gambar Tidak Valid</strong><br>{reason}</div>', unsafe_allow_html=True)
        st.markdown("""
        **Rekomendasi:**
        - Foto batu secara langsung dengan jarak dekat
        - Gunakan pencahayaan alami atau cukup terang
        - Hindari objek lain yang mendominasi frame
        - Pastikan fokus kamera pada tekstur batu
        """)
    else:
        st.markdown(f'<div class="success-card"><strong>Gambar Valid</strong><br>{reason}</div>', unsafe_allow_html=True)
        
        # Tombol klasifikasi
        if st.button("Mulai Klasifikasi", type="primary"):
            with st.spinner("Sedang menganalisis gambar..."):
                # Enhancement untuk prediksi lebih akurat
                img_enhanced = image.filter(ImageFilter.SHARPEN)
                enhancer = ImageEnhance.Contrast(img_enhanced)
                img_enhanced = enhancer.enhance(1.2)
                
                # Prediksi
                predictions = predict_image(interpreter, input_details, output_details, img_enhanced, use_real_model)
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else list(DESKRIPSI_KELAS.keys())[pred_idx] if pred_idx < len(DESKRIPSI_KELAS) else "Unknown"
                confidence = float(predictions[pred_idx])
                
                # Top 3 predictions
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = []
                for i in top_3_idx:
                    cls = class_names[i] if i < len(class_names) else list(DESKRIPSI_KELAS.keys())[i] if i < len(DESKRIPSI_KELAS) else "Unknown"
                    top_3.append((cls, float(predictions[i])))
            
            # Tampilkan hasil
            st.markdown("### Hasil Klasifikasi")
            
            if confidence >= CONFIDENCE_THRESHOLD or not use_real_model:
                # Hasil berhasil (atau mode demo)
                conf_class = "confidence-high" if confidence > 0.8 else "confidence-medium"
                
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="color: #2a5298; margin-top: 0; margin-bottom: 0.5rem;">{pred_class}</h3>
                    <p class="{conf_class}" style="margin: 0 0 1rem 0;">{confidence:.1%} Confidence</p>
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 1rem 0;">
                    <p style="margin: 0 0 0.5rem 0;"><strong>Deskripsi:</strong></p>
                    <p style="text-align: justify; margin: 0; color: #555;">{DESKRIPSI_KELAS.get(pred_class, 'Deskripsi tidak tersedia untuk kelas ini.')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if not use_real_model:
                    st.info("ℹ️ **Mode Demo**: Hasil ini adalah prediksi simulasi. Install TensorFlow untuk hasil akurat.")
                
                # Top 3 predictions
                st.markdown("### Prediksi Lainnya")
                for i, (cls, conf) in enumerate(top_3, 1):
                    bar_width = min(int(conf * 100), 100)
                    st.markdown(f"""
                    <div style="margin: 0.75rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem; font-size: 0.9rem;">
                            <span>{i}. {cls}</span>
                            <span style="color: #666;">{conf:.1%}</span>
                        </div>
                        <div style="background: #e0e0e0; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                        width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Grafik distribusi probabilitas
                st.markdown("### Distribusi Probabilitas")
                chart_data = {
                    "Kelas": [cls for cls, _ in top_3],
                    "Probabilitas": [conf for _, conf in top_3]
                }
                st.bar_chart(chart_data, x="Kelas", y="Probabilitas", height=250)
                
                # Download laporan
                html_content = buat_html_hasil(
                    gambar.name if hasattr(gambar, 'name') else f"foto_{int(time.time())}.jpg",
                    pred_class,
                    confidence,
                    top_3,
                    DESKRIPSI_KELAS.get(pred_class, "")
                )
                
                st.download_button(
                    label="Download Laporan (HTML)",
                    data=html_content,
                    file_name=f"klasifikasi_{pred_class}_{int(time.time())}.html",
                    mime="text/html",
                    use_container_width=True,
                    type="primary"
                )
                
                st.info("💡 **Tips:** Buka file HTML dengan browser untuk melihat laporan lengkap. Dari browser, Anda bisa mencetak (Ctrl+P) dan menyimpan sebagai PDF.")
                
            else:
                # Confidence rendah
                st.markdown(f"""
                <div class="warning-card">
                    <strong>Hasil Kurang Yakin</strong><br>
                    Model memberikan confidence {confidence:.1%} yang berada di bawah threshold {CONFIDENCE_THRESHOLD:.0%}.
                    <br><br>
                    <strong>Prediksi terbaik:</strong> {pred_class}
                    <br><br>
                    <em>Saran: Coba ambil foto dengan pencahayaan lebih baik atau sudut yang lebih jelas.</em>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Semua Prediksi")
                for i, (cls, conf) in enumerate(top_3, 1):
                    st.write(f"{i}. {cls}: {conf:.1%}")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem; font-size: 0.85rem;">
    <p>Aplikasi Klasifikasi Batu Megalitikum</p>
    <p>Berdasarkan Penelitian Deep Learning</p>
</div>
""", unsafe_allow_html=True)
