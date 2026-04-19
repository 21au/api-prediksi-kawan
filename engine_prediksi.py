import pandas as pd
import numpy as np
import logging
from datetime import datetime
from prophet import Prophet
from supabase import create_client, Client
from pygrowup import Calculator
import warnings
import os
from dotenv import load_dotenv

# Menonaktifkan peringatan yang tidak perlu agar log API tetap bersih
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# =====================================================================
# 1. KONFIGURASI SUPABASE & WHO/KEMENKES Z-SCORE
# =====================================================================
load_dotenv() 

# AMAN: Mengambil Key dari file .env agar tidak bocor
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Pengecekan jika file .env lupa dibuat/dibaca
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL atau SUPABASE_KEY tidak ditemukan di file .env!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# pygrowup menggunakan data LMS WHO 2006, yang secara resmi 
# diadopsi oleh Kemenkes RI dalam Permenkes No 2 Tahun 2020.
cg = Calculator(adjust_height_data=False, adjust_weight_scores=False)


# =====================================================================
# 2. FUNGSI KLASIFIKASI (STANDAR PERMENKES NO. 2 TAHUN 2020)
# =====================================================================
def klasifikasi_status_gizi(z_score, indikator):
    """
    Mengklasifikasikan status gizi berdasarkan nilai Z-Score dan indikator
    merujuk pada standar Permenkes No. 2 Tahun 2020 (Halaman 12-13).
    """
    if z_score is None: 
        return "Tidak Diketahui"
    
    # Standar BB/U (Berat Badan menurut Umur)
    if indikator == 'BB/U': 
        if z_score < -3.0: 
            return "Berat Badan Sangat Kurang (Severely Underweight)"
        elif -3.0 <= z_score < -2.0: 
            return "Berat Badan Kurang (Underweight)"
        elif -2.0 <= z_score <= 1.0: 
            return "Berat Badan Normal"
        else: 
            return "Risiko Berat Badan Lebih"
        
    # Standar TB/U atau PB/U (Tinggi/Panjang Badan menurut Umur)
    elif indikator == 'TB/U': 
        if z_score < -3.0: 
            return "Sangat Pendek (Severely Stunted)"
        elif -3.0 <= z_score < -2.0: 
            return "Pendek (Stunted)"
        elif -2.0 <= z_score <= 3.0: 
            return "Normal"
        else: 
            return "Tinggi"
        
    return "Normal"


# =====================================================================
# 3. FUNGSI UTAMA PREDIKSI (MAKSIMAL 5 BULAN KE DEPAN)
# =====================================================================
def proses_prediksi_pertumbuhan():
    """
    Fungsi utama untuk mengambil data, menjalankan model Prophet,
    menghitung Z-Score, dan menyimpannya kembali ke database.
    """
    try:
        # Ambil Data dari Supabase
        res_anak = supabase.table('anak').select("*").execute()
        df_anak = pd.DataFrame(res_anak.data)

        res_tumbuh = supabase.table('pertumbuhan').select("*").execute()
        df_pertumbuhan = pd.DataFrame(res_tumbuh.data)
        
    except Exception as e:
        return {"status": "error", "message": f"Error koneksi Supabase: {e}"}

    if df_anak.empty or df_pertumbuhan.empty:
        return {"status": "success", "message": "Data anak atau pertumbuhan masih kosong."}

    # Format Tanggal
    df_pertumbuhan['tanggal_pengukuran'] = pd.to_datetime(df_pertumbuhan['tanggal_pengukuran'])
    df_anak['tanggal_lahir'] = pd.to_datetime(df_anak['tanggal_lahir'])

    # Gabung Data
    df = pd.merge(df_pertumbuhan, df_anak[['id', 'nama', 'tanggal_lahir', 'jenis_kelamin']], left_on='anak_id', right_on='id')

    # Standarisasi Gender untuk pygrowup (WHO)
    df['gender_who'] = df['jenis_kelamin'].str.upper().map({
        'L': 'M', 'P': 'F', 'LAKI-LAKI': 'M', 'PEREMPUAN': 'F'
    }).fillna('M')

    indikator_map = {'berat_badan': 'wfa', 'tinggi_badan': 'hfa'}
    hasil_db = []

    # Proses per Anak
    for anak_id in df['anak_id'].unique():
        df_s = df[df['anak_id'] == anak_id].sort_values('tanggal_pengukuran')
        nama_anak = df_s['nama'].iloc[0]
        gender = df_s['gender_who'].iloc[0]
        tgl_lahir = df_s['tanggal_lahir'].iloc[0]

        print(f"\nAnalisis: {nama_anak} (Jumlah Data: {len(df_s)})")

        # SYARAT UTAMA: Minimal 3 Data untuk Prophet
        if len(df_s) < 3:
            print(f"⚠️ Skip! {nama_anak} butuh minimal 3 data agar prediksi akurat.")
            continue

        for m_db, m_who in indikator_map.items():
            if m_db not in df_s.columns or df_s[m_db].isnull().all(): continue

            try:
                df_p = df_s[['tanggal_pengukuran', m_db]].rename(columns={'tanggal_pengukuran':'ds', m_db:'y'})
                
                # --- PROPHET SETUP ---
                model = Prophet(
                    yearly_seasonality=False, 
                    weekly_seasonality=False, 
                    daily_seasonality=False,
                    changepoint_prior_scale=0.01 
                )
                model.fit(df_p)
                
                # Setting periods=5 untuk memprediksi 5 bulan ke depan secara langsung
                future = model.make_future_dataframe(periods=5, freq='30D')
                forecast = model.predict(future)
                
                # Mengambil 5 data terakhir (hasil prediksi masa depan)
                future_predictions = forecast.tail(5)
                
            except Exception as e:
                print(f"   [!] Model Prophet gagal memproses indikator {m_db}: {e}")
                continue

            # Looping untuk memproses hasil prediksi (Bulan 1 sampai 5)
            bulan_ke = 1
            for _, row in future_predictions.iterrows():
                val_pred = round(row['yhat'], 2)
                tgl_pred = row['ds']
                
                # Hitung Umur saat tanggal prediksi
                umur_bln = (tgl_pred - tgl_lahir).days / 30.4375
                
                try:
                    # Hitung Z-Score menggunakan pygrowup
                    if m_who == 'wfa':
                        z = cg.wfa(measurement=float(val_pred), age_in_months=float(umur_bln), sex=gender)
                    elif m_who == 'hfa':
                        z = cg.lhfa(measurement=float(val_pred), age_in_months=float(umur_bln), sex=gender)
                    else:
                        z = 0.0

                    # Klasifikasi Status Gizi
                    if z is not None:
                        lbl = 'BB/U' if m_who == 'wfa' else 'TB/U'
                        status = klasifikasi_status_gizi(z, lbl)
                    else:
                        z, status = 0.0, "Luar Jangkauan Umur (> 5 Tahun)"
                        
                except Exception as e:
                    z, status = 0.0, "Gagal Hitung Z-Score"

                # Masukkan ke list untuk disimpan ke DB
                hasil_db.append({
                    'anak_id': str(anak_id), # AMAN: Menggunakan str() untuk mencegah crash pada UUID
                    'metrik': m_db,
                    'tanggal_prediksi': tgl_pred.strftime('%Y-%m-%d'),
                    'nilai_prediksi': float(val_pred),
                    'z_score': float(round(z, 2)) if z else 0.0,
                    'status_gizi': status,
                    'created_at': datetime.now().isoformat()
                })
                
                print(f" > {m_db} (Bulan +{bulan_ke}): Pred={val_pred} | {status} (Z-Score: {round(z,2) if z else 0})")
                bulan_ke += 1

    # =====================================================================
    # 4. PUSH HASIL KE SUPABASE
    # =====================================================================
    if hasil_db:
        try:
            # Hapus data prediksi lama per anak agar tidak menumpuk saat dijalankan ulang
            list_anak_id = list(set([d['anak_id'] for d in hasil_db]))
            supabase.table('prediksi_pertumbuhan').delete().in_('anak_id', list_anak_id).execute()
            
            # Masukkan seluruh 5 data prediksi baru
            supabase.table('prediksi_pertumbuhan').insert(hasil_db).execute()
            msg = f"Berhasil menyimpan {len(hasil_db)} baris prediksi ke database."
            print(f"\n✅ {msg}")
            return {"status": "success", "message": msg, "data_processed": len(hasil_db)}
        except Exception as e:
            error_msg = f"Gagal menyimpan ke database: {e}"
            print(f"\n❌ {error_msg}")
            return {"status": "error", "message": error_msg}
    else:
        msg = "Tidak ada data yang memenuhi syarat untuk diprediksi (minimal 3 data)."
        print(f"\n💡 {msg}")
        return {"status": "success", "message": msg}


# =====================================================================
# BLOK EKSEKUSI 
# =====================================================================
# if __name__ == "__main__":
#     print("--- Memulai Proses Prediksi ---")
#     hasil = proses_prediksi_pertumbuhan()
#     print("--- Selesai ---")
#     print("Response API:", hasil)