import pandas as pd
from datetime import datetime
from prophet import Prophet
from supabase import create_client, Client
from pygrowup import Calculator
import os
import warnings
from dotenv import load_dotenv
import logging

# Menghilangkan pesan log dari library yang tidak perlu agar terminal bersih
warnings.filterwarnings('ignore')
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# 1. KONFIGURASI
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- FUNGSI KLASIFIKASI WHO ---
def klasifikasi_status_gizi(z_score, indikator):
    """Mengklasifikasikan status gizi berdasarkan standar Z-Score WHO"""
    if pd.isna(z_score) or z_score is None: 
        return "Tidak Diketahui"
    
    if indikator == 'BB/U': # Berat Badan menurut Umur
        if z_score < -3.0: return "Sangat Kurang (Gizi Buruk)"
        elif -3.0 <= z_score < -2.0: return "Kurang (Gizi Kurang)"
        elif -2.0 <= z_score <= 1.0: return "Normal"
        else: return "Risiko Lebih"
        
    elif indikator == 'TB/U': # Tinggi Badan menurut Umur
        if z_score < -3.0: return "Sangat Pendek (Severely Stunted)"
        elif -3.0 <= z_score < -2.0: return "Pendek (Stunted)"
        elif -2.0 <= z_score <= 3.0: return "Normal"
        else: return "Tinggi"
        
    return "Normal"

# --- CORE ENGINE ---
def jalankan_mesin():
    """
    Fungsi utama mesin. Didesain me-return Dictionary agar 
    nantinya sangat mudah dicolokkan ke FastAPI / Flask.
    """
    print("🚀 Memulai Mesin Prediksi Kawan Tumbuh...")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"status": "error", "message": "SUPABASE_URL atau KEY tidak ditemukan di file .env"}

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # adjust_height_data=False agar tidak otomatis memotong tinggi anak di bawah 2 tahun
    cg = Calculator(adjust_height_data=False, adjust_weight_scores=False)

    try:
        # 2. AMBIL DATA DARI SUPABASE
        res_anak = supabase.table('anak').select("*").execute()
        res_tumbuh = supabase.table('pertumbuhan').select("*").execute()
        
        df_anak = pd.DataFrame(res_anak.data)
        df_tumbuh = pd.DataFrame(res_tumbuh.data)

        if df_anak.empty or df_tumbuh.empty:
            return {"status": "success", "message": "Data di Supabase masih kosong. Tidak ada yang diproses."}

        # 3. PRE-PROCESSING (Pembersihan Data)
        df_tumbuh['tanggal_pengukuran'] = pd.to_datetime(df_tumbuh['tanggal_pengukuran'])
        df_anak['tanggal_lahir'] = pd.to_datetime(df_anak['tanggal_lahir'])
        
        # Filter data tidak wajar (misal berat <= 0) agar AI tidak rusak
        df_tumbuh = df_tumbuh[(df_tumbuh['berat_badan'] > 0) & (df_tumbuh['tinggi_badan'] > 0)]

        df = pd.merge(df_tumbuh, df_anak[['id', 'nama', 'tanggal_lahir', 'jenis_kelamin']], left_on='anak_id', right_on='id')
        df['gender_who'] = df['jenis_kelamin'].str.upper().map({'L': 'M', 'P': 'F', 'LAKI-LAKI': 'M', 'PEREMPUAN': 'F'}).fillna('M')

        hasil_db = []
        anak_diproses = 0
        
        # 4. PROSES PREDIKSI PER ANAK
        for anak_id in df['anak_id'].unique():
            df_s = df[df['anak_id'] == anak_id].sort_values('tanggal_pengukuran')
            nama_anak = df_s['nama'].iloc[0]
            
            if len(df_s) < 2:
                print(f"⚠️ Skip {nama_anak}: Data kurang (minimal 2).")
                continue

            anak_diproses += 1
            print(f"⚙️ Memproses prediksi untuk anak: {nama_anak}...")

            for m_db, m_who in {'berat_badan': 'wfa', 'tinggi_badan': 'hfa'}.items():
                try:
                    df_p = df_s[['tanggal_pengukuran', m_db]].rename(columns={'tanggal_pengukuran':'ds', m_db:'y'})
                    
                    # Model AI Prophet (Setting anti-overfitting untuk data sedikit)
                    model = Prophet(
                        yearly_seasonality=False, 
                        weekly_seasonality=False, 
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05 # Mengurangi lonjakan tajam jika ada data aneh
                    )
                    model.fit(df_p)
                    
                    # Prediksi untuk 5 bulan ke depan
                    future = model.make_future_dataframe(periods=5, freq='30D')
                    forecast = model.predict(future)
                    
                    for _, row in forecast.tail(5).iterrows():
                        # Perhitungan Umur
                        hari_hidup = (row['ds'] - df_s['tanggal_lahir'].iloc[0]).days
                        umur_bln = max(0, hari_hidup / 30.4375) # Pastikan umur tidak minus
                        
                        # Jika umur lebih dari batas WHO (biasanya 60-120 bulan), lewati z-score
                        if umur_bln > 60 and m_who == 'wfa': 
                            continue 
                            
                        # Hitung Z-Score dengan PyGrowUp
                        try:
                            if m_who == 'wfa':
                                z = cg.wfa(float(row['yhat']), umur_bln, df_s['gender_who'].iloc[0])
                                indikator_nama = 'BB/U'
                            else:
                                z = cg.lhfa(float(row['yhat']), umur_bln, df_s['gender_who'].iloc[0])
                                indikator_nama = 'TB/U'
                        except Exception as e_z:
                            z = 0.0 # Fallback jika pygrowup gagal menghitung
                            indikator_nama = 'BB/U'

                        status_akhir = klasifikasi_status_gizi(z, indikator_nama)
                        
                        hasil_db.append({
                            'anak_id': str(anak_id),
                            'metrik': m_db,
                            'tanggal_prediksi': row['ds'].strftime('%Y-%m-%d'),
                            'nilai_prediksi': float(round(row['yhat'], 2)),
                            'z_score': float(round(z, 2)) if z else 0.0,
                            'status_gizi': status_akhir, 
                            'created_at': datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    print(f"❌ Error model pada {nama_anak} ({m_db}): {e}")
                    continue # Lanjut ke metrik/anak berikutnya tanpa mematikan sistem

        # 5. SIMPAN HASIL KE SUPABASE
        if hasil_db:
            list_id = list(set([d['anak_id'] for d in hasil_db]))
            
            # Hapus prediksi lama agar tidak dobel
            supabase.table('prediksi_pertumbuhan').delete().in_('anak_id', list_id).execute()
            # Masukkan yang baru
            supabase.table('prediksi_pertumbuhan').insert(hasil_db).execute()
            
            return {
                "status": "success", 
                "message": f"Berhasil memperbarui {len(hasil_db)} baris prediksi untuk {anak_diproses} anak."
            }
        else:
            return {"status": "success", "message": "Tidak ada data prediksi baru yang memenuhi syarat."}

    except Exception as e:
        return {"status": "error", "message": f"Terjadi Kesalahan Global: {str(e)}"}

# === BLOK EKSEKUSI LOKAL ===
if __name__ == "__main__":
    # Jika dijalankan biasa (python engine_prediksi.py), hasil JSON akan di-print
    hasil_eksekusi = jalankan_mesin()
    print("\n--- HASIL AKHIR ---")
    print(hasil_eksekusi)