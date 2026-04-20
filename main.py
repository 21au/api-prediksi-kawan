from fastapi import FastAPI, BackgroundTasks
from engine_prediksi import jalankan_mesin

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Mesin Prediksi Kawan Tumbuh Aktif"}

# Ini adalah 'pintu' yang akan dipanggil oleh Supabase
@app.post("/trigger-predict")
def trigger_predict(background_tasks: BackgroundTasks):
    # Pakai background_tasks supaya Supabase tidak kelamaan nunggu 
    # proses AI Prophet yang berat
    background_tasks.add_task(jalankan_mesin)
    return {"message": "Proses prediksi dimulai di latar belakang"}