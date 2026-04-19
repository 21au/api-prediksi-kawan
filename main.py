from flask import Flask, jsonify
import os

# Import fungsi prediksi dari file engine_prediksi.py milikmu
from engine_prediksi import proses_prediksi_pertumbuhan

app = Flask(__name__)

# Halaman depan untuk cek apakah server hidup
@app.route('/', methods=['GET'])
def home():
    return "Server Mesin Prediksi Kawan Tumbuh Aktif dan Berjalan!"

# Alamat ini yang akan dipanggil otomatis oleh Supabase
@app.route('/webhook-prediksi', methods=['POST'])
def webhook_prediksi():
    try:
        print("Menerima trigger dari Supabase! Memulai prediksi otomatis...")
        # Menjalankan fungsi prediksimu
        hasil = proses_prediksi_pertumbuhan()
        return jsonify(hasil), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Wajib seperti ini agar bisa berjalan di hosting cloud (Render)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)