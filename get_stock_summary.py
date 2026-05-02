# pip install curl_cffi --upgrade

from curl_cffi.requests import Session
import json
from datetime import datetime, timedelta
from typing import Any, List
from pathlib import Path
import time
import random

URL_API = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
URL_PAGE = "https://www.idx.co.id/en/market-data/trading-summary/stock-summary/"

# Buat folder 'data' dan 'backup' jika belum ada
data_folder = Path("data")
backup_folder = Path("backup")
data_folder.mkdir(exist_ok=True)
backup_folder.mkdir(exist_ok=True)

print("=" * 60)
print("DOWNLOAD DATA SAHAM IDX - HISTORICAL DATA")
print("=" * 60)
print("1. Download hari ini (tekan ENTER)")
print("2. Download 1 tanggal spesifik")
print("3. Download beberapa tahun kebelakang (otomatis)")
print("=" * 60)
choice = input("Pilih [1/2/3 atau Enter]: ").strip()

# Tentukan folder tujuan
target_folder = data_folder  # Default ke data folder

if choice == "" or choice == "1":
    # Mode default - tanggal hari ini
    today = datetime.now()
    date_input = today.strftime("%Y%m%d")
    print(f"\n📅 Download data hari ini: {today.strftime('%d %B %Y')} ({date_input})")
    dates_to_download: List[str] = [date_input]
    target_folder = data_folder

elif choice == "2":
    # Mode manual - 1 tanggal
    date_input = input("Masukkan tanggal (format: YYYYMMDD, contoh: 20260209): ").strip()
    
    # Validasi format tanggal
    try:
        datetime.strptime(date_input, "%Y%m%d")
        if len(date_input) != 8:
            raise ValueError("Format harus 8 digit")
    except ValueError as e:
        print(f"❌ Error: Format tanggal salah! Gunakan format YYYYMMDD (8 digit)")
        print(f"   Contoh: 20260209 untuk tanggal 9 Februari 2026")
        exit(1)
    
    dates_to_download: List[str] = [date_input]
    target_folder = data_folder

elif choice == "3":
    # Mode otomatis - input tahun yang diinginkan, simpan ke BACKUP folder
    try:
        years_input = int(input("Berapa tahun kebelakang yang mau didownload? (contoh: 3 untuk 3 tahun): ").strip())
        if years_input <= 0:
            print("❌ Tahun harus lebih dari 0!")
            exit(1)
    except ValueError:
        print("❌ Input harus angka!")
        exit(1)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_input*365)  # Custom tahun
    
    print(f"\n📅 Download data dari {start_date.strftime('%Y-%m-%d')} sampai {end_date.strftime('%Y-%m-%d')}")
    estimasi_hari = years_input * 252  # 252 hari bursa per tahun (rata-rata)
    print(f"⏳ Estimasi: ~{estimasi_hari} hari bursa (akan skip weekend & file yang sudah ada)")
    print(f"📁 File akan disimpan ke folder: {backup_folder.absolute()}")
    confirm = input("Lanjutkan? [y/n]: ").strip().lower()
    
    if confirm != 'y':
        print("Dibatalkan.")
        exit(0)
    
    # Generate list tanggal (skip weekend)
    dates_to_download: List[str] = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekend (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            dates_to_download.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    print(f"\n📊 Total hari kerja: {len(dates_to_download)} hari")
    target_folder = backup_folder  # Simpan ke backup folder untuk opsi 3

else:
    print("❌ Pilihan tidak valid!")
    exit(1)

# Download data
success_count = 0
skip_count = 0
error_count = 0
total = len(dates_to_download)

print(f"\n🚀 Memulai download...\n")
print("💡 Tips: IDX punya bot detection. Script akan jalan LAMBAT untuk avoid ban.\n")

# Coba browser impersonation terbaru
browser_versions = ["chrome120", "chrome124", "chrome123", "safari17_0"]
session = None

for browser in browser_versions:
    try:
        print(f"⚡ Mencoba browser: {browser}...")
        session = Session(impersonate=browser)  # type: ignore[misc]
        print(f"   ✅ Berhasil: {browser}\n")
        break
    except Exception as e:
        print(f"   ❌ Gagal: {browser}")
        continue

if session is None:
    print("❌ Semua browser impersonation gagal!")
    print("💡 Solusi:")
    print("   1. pip install curl_cffi --upgrade")
    print("   2. pip uninstall curl_cffi -y && pip install curl_cffi")
    exit(1)

with session as s:  # type: ignore[misc]
    # Warming up session sekali saja
    print("⚡ Warming up session...")
    try:
        warmup_resp = s.get(URL_PAGE, timeout=30)
        print(f"   ✅ Session ready (status: {warmup_resp.status_code})")
        time.sleep(2)  # Kasih jeda setelah warmup
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
        print("   Melanjutkan download...")
    
    for idx, date_str in enumerate(dates_to_download, 1):
        # Check jika file sudah ada
        filename = f"idx_stock_{date_str}.json"
        filepath = target_folder / filename
        data_path = data_folder / filename
        backup_path = backup_folder / filename
        
        if data_path.exists() or backup_path.exists():
            print(f"[{idx}/{total}] ⏭️  {date_str} - sudah ada, skip")
            skip_count += 1
            continue
        
        retry_count = 0
        max_retries = 3
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Download data
                resp = s.get(
                    URL_API,
                    params={"length": 9999, "start": 0, "date": date_str},
                    headers={
                        "accept": "application/json, text/plain, */*",
                        "accept-language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                        "cache-control": "no-cache",
                        "pragma": "no-cache",
                        "referer": URL_PAGE,
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "same-origin",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Windows"',
                    },
                    timeout=60,
                )
                
                # Check HTTP status
                if resp.status_code == 429:
                    wait_time = random.uniform(15, 25)
                    print(f"[{idx}/{total}] ⚠️  {date_str} - Rate limit! Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                elif resp.status_code == 403:
                    wait_time = random.uniform(20, 40) * (retry_count + 1)
                    print(f"[{idx}/{total}] 🚫 {date_str} - Bot detected! Retry {retry_count+1}/{max_retries}...")
                    print(f"    ⏳ Waiting {wait_time:.1f}s (longer delay untuk avoid ban)...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                elif resp.status_code >= 500:
                    print(f"[{idx}/{total}] ⚠️  {date_str} - Server error ({resp.status_code}). Retry {retry_count+1}/{max_retries}...")
                    time.sleep(5)
                    retry_count += 1
                    continue
                
                resp.raise_for_status()
                data: dict[str, Any] = json.loads(resp.text)
                
                # Jika tidak ada data (hari libur), skip
                if data.get("recordsTotal", 0) == 0:
                    print(f"[{idx}/{total}] 🚫 {date_str} - tidak ada data (libur)")
                    skip_count += 1
                    time.sleep(1)  # Delay kecil
                    break
                
                # Save ke data folder
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"[{idx}/{total}] ✅ {date_str} - {data.get('recordsTotal', '?')} saham")
                success_count += 1
                success = True
                
                # Delay untuk avoid rate limit - PENTING!
                base_delay = random.uniform(3, 7)  # Random 3-7 detik
                time.sleep(base_delay)
                
                # Extra cooling setiap 5 request
                if idx % 5 == 0:
                    cooling_time = random.uniform(10, 15)
                    print(f"    💤 Cooling down {cooling_time:.1f} detik (avoid bot detection)...")
                    time.sleep(cooling_time)
                
                # Super cooling setiap 20 request
                if idx % 20 == 0:
                    super_cooling = random.uniform(30, 45)
                    print(f"    🧊 SUPER Cooling {super_cooling:.1f} detik (prevent IP ban)...")
                    time.sleep(super_cooling)
                    
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Detailed error reporting
                if hasattr(e, 'response') and e.response is not None:  # type: ignore
                    status_code = e.response.status_code  # type: ignore
                    print(f"[{idx}/{total}] ❌ {date_str} - HTTP {status_code}: {error_msg}")
                else:
                    print(f"[{idx}/{total}] ❌ {date_str} - {error_type}: {error_msg}")
                
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 3
                    print(f"    🔄 Retry {retry_count}/{max_retries} dalam {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    ⛔ Max retries reached, skipping...")
                    error_count += 1
                    time.sleep(2)

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"✅ Berhasil: {success_count}")
print(f"⏭️  Dilewati: {skip_count}")
print(f"❌ Error: {error_count}")
print(f"📁 Lokasi: {target_folder.absolute()}")
print("=" * 60)