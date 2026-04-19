from flask import Flask, request, jsonify
import os
from supabase import create_client, Client
# Import fungsi prediksi milikmu
from engine_prediksi import proses_prediksi_pertumbuhan

app = Flask(__name__)

# --- KONFIGURASI SUPABASE ---
# Mengambil URL dan KEY yang sudah kamu masukkan di Settings -> Secrets Hugging Face
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@app.route('/', methods=['GET'])
def home():
    return "Server Mesin Prediksi Kawan Tumbuh Aktif!"

@app.route('/webhook-prediksi', methods=['POST'])
def webhook_prediksi():
    try:
        # 1. Menangkap data yang dikirim oleh Webhook Supabase
        payload = request.json
        print(f"Menerima trigger! Data: {payload}")

        # 2. Mengambil detail record baru dari tabel 'pertumbuhan'
        # Supabase Webhook mengirim data dalam struktur: {'record': {...}}
        record_baru = payload.get('record', {})
        
        if not record_baru:
            return jsonify({"status": "error", "message": "Data record tidak ditemukan"}), 400

        # 3. Jalankan fungsi prediksi kamu
        # Pastikan fungsi 'proses_prediksi_pertumbuhan' di file sebelah bisa menerima parameter ini jika perlu
        # Atau jika fungsinya otomatis membaca DB, pastikan logic-nya sesuai
        hasil_prediksi = proses_prediksi_pertumbuhan()

        # 4. Jika hasil_prediksi adalah LIST (karena biasanya prediksi banyak baris), 
        # kita masukkan semuanya ke tabel 'prediksi_pertumbuhan'
        if hasil_prediksi:
            # Perintah untuk simpan ke tabel yang kamu tunjukkan di gambar tadi
            response = supabase.table("prediksi_pertumbuhan").insert(hasil_prediksi).execute()
            print("Hasil prediksi berhasil disimpan ke Supabase!")
            return jsonify({"status": "success", "data": response.data}), 200
        else:
            return jsonify({"status": "warning", "message": "Tidak ada hasil prediksi yang dihasilkan"}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Port 7860 adalah standar untuk Hugging Face Spaces
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)