# ozleyisyavuz-power-demand-forecast-tr
# Renewable Generation Forecast TR — Rüzgâr/Güneş Üretimi + Belirsizlik (P10/P50/P90) + FastAPI + CI

[![CI](https://github.com/<kullanici>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<kullanici>/<repo>/actions/workflows/ci.yml)

> **Amaç:** Yenilenebilir enerji üretimi (rüzgâr + güneş) için saatlik tahmin üreten ve belirsizliği yönetmek amacıyla **P10 / P50 / P90** senaryolarını döndüren, API üzerinden kullanılabilir ve CI ile doğrulanmış uçtan uca bir demo sistem geliştirmek.

---

## Neden bu proje?
Yenilenebilir üretim (özellikle rüzgâr ve güneş) **hava koşullarına bağlı** olduğu için doğal olarak **değişken ve belirsizdir**. Gerçek operasyonlarda çoğu zaman tek bir sayı (nokta tahmin) yeterli olmaz.

Bu yüzden bu projede tahmin çıktısını üç senaryo olarak sunuyoruz:

- **P10 (kötümser/düşük senaryo):** Üretimin düşük gelme ihtimalini temsil eder  
- **P50 (en olası):** En gerçekçi/orta senaryo  
- **P90 (iyimser/yüksek senaryo):** Üretimin yüksek gelme ihtimalini temsil eder  

Bu yaklaşım; planlama, risk yönetimi, rezerv ihtiyacı ve ticaret kararları için daha anlamlı bir temel sağlar.

---

Bu çalışma “sadece model eğitmek” değil; küçük ölçekte **üretime yakın bir ML sistemi** örneği sunar.

### 1) Veri üretimi (sentetik/demo)
`make_dataset.py`, saatlik bir veri seti üretir:
- `wind_speed_mps` → rüzgâr hızı (m/s)
- `ghi_wm2` → güneş ışınımı (irradiance) benzeri gösterge (W/m²)
- `temperature_c` → sıcaklık (°C)
- `generation_mw` → hedef değişken (rüzgâr+güneş toplam üretim MW)

Çıktı:
- `data/processed/renewables.csv`

### 2) Modelleme: P10 / P50 / P90 (quantile regression)
`train.py`, scikit-learn ile 3 ayrı model eğitir:
- `models/q10.joblib` → P10
- `models/q50.joblib` → P50
- `models/q90.joblib` → P90

Eğitim sonunda aşağıdaki metrikler yazdırılır:
- **MAE (MW)**
- **MAPE (%)**
- **P10–P90 kapsama oranı (coverage %)**  
  (Gerçek değerlerin ne kadarının P10 ile P90 arasında kaldığı)

### 3) API ile servisleştirme (FastAPI)
`main.py`, eğitilen modelleri bir REST servisi olarak yayınlar:
- `POST /predict` → `{p10_mw, p50_mw, p90_mw}`
- `GET /health` → servis çalışıyor mu?

Swagger (etkileşimli dokümantasyon):
- `/docs`

### 4) Test + CI (GitHub Actions)
Repo’da uçtan uca test vardır:
- veri üret → eğitim → API → `/predict` isteği → çıktı doğrulama

GitHub Actions CI her commit/PR’da testleri çalıştırır ve “bozulmayı” engeller.

---

## Proje Yapısı (Repository Layout)
```text
.github/workflows/ci.yml
requirements.txt
pytest.ini
README.md

src/renewable_generation_forecast/
  app/main.py
  data/make_dataset.py
  models/train.py

tests/test_end_to_end.py
