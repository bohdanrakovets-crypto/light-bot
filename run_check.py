import asyncio
import numpy as np
import cv2
import pytesseract
import re
import requests
import json
import os
import hashlib
from datetime import datetime, date
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from aiogram import Bot
from aiogram.types import BufferedInputFile

# --- НАЛАШТУВАННЯ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = os.getenv("GROUP_ID") 
SITE_URL = "https://voe-poweron.inneti.net/schedule_queues"

pytesseract.pytesseract.tesseract_cmd = 'tesseract'

LOWER_BLUE = np.array([80, 60, 40])
UPPER_BLUE = np.array([255, 180, 120])
TARGET_QUEUE_INDEX = 4 # Черга 3.1
STATE_FILE = "state.json"

# --- ФУНКЦІЇ ---

def extract_queue_pixels(img):
    """
    Вирізає з картинки ТІЛЬКИ смужки, що стосуються черги 3.1.
    Повертає байти цих смужок для хешування.
    """
    h, w, _ = img.shape
    rows_total = 12
    
    # Координати блоків (ті самі, що в аналізі)
    top_y_start = int(h * 0.19)
    top_y_end = int(h * 0.51)
    
    bottom_y_start = int(h * 0.58)
    bottom_y_end = int(h * 0.90)

    # 1. Вирізаємо смужку з верхнього блоку
    t_block_h = top_y_end - top_y_start
    t_row_h = t_block_h / rows_total
    # Координати конкретного рядка
    t_r_start = int(top_y_start + (TARGET_QUEUE_INDEX * t_row_h))
    t_r_end = int(t_r_start + t_row_h)
    
    row_top = img[t_r_start:t_r_end, :]

    # 2. Вирізаємо смужку з нижнього блоку
    b_block_h = bottom_y_end - bottom_y_start
    b_row_h = b_block_h / rows_total
    b_r_start = int(bottom_y_start + (TARGET_QUEUE_INDEX * b_row_h))
    b_r_end = int(b_r_start + b_row_h)
    
    row_bottom = img[b_r_start:b_r_end, :]

    # 3. Склеюємо їх разом (вертикально)
    combined_rows = np.vstack((row_top, row_bottom))
    
    # Повертаємо як набір байтів
    return combined_rows.tobytes()

def calculate_hash(data_bytes):
    return hashlib.md5(data_bytes).hexdigest()

def get_image_links_headless():
    print("🚀 Selenium: Start...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    found_urls = []
    try:
        driver.get(SITE_URL)
        try:
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "img")))
        except: pass
        
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            src = img.get_attribute("src")
            if src and (("GPV" in src) or ("media" in src and ("png" in src or "jpg" in src))):
                 found_urls.append(src)
    except Exception as e:
        print(f"Selenium Error: {e}")
    finally:
        driver.quit()
    return list(set(found_urls))

def parse_date_only(img):
    try:
        h, w, _ = img.shape
        header_crop = img[0:int(h*0.15), 0:int(w*0.50)]
        gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, lang='ukr+eng')
        dm = re.findall(r'(\d{2})\.(\d{2})\.(\d{4})', text)
        if dm:
            return datetime.strptime(f"{dm[0][0]}.{dm[0][1]}.{dm[0][2]}", "%d.%m.%Y").date()
    except: pass
    return None

def analyze_schedule_image(img):
    height, width, _ = img.shape
    debug_img = img.copy()
    rows_total = 12
    top_y_start = int(height * 0.19)
    top_y_end = int(height * 0.51)
    bottom_y_start = int(height * 0.58)
    bottom_y_end = int(height * 0.90)
    outage_intervals = []

    def scan_block(y_start, y_end, hour_offset):
        block_h = y_end - y_start
        row_h = block_h / rows_total
        y_center = int(y_start + (TARGET_QUEUE_INDEX * row_h) + (row_h / 2))
        
        cv2.line(debug_img, (0, y_center), (width, y_center), (0, 255, 0), 2)
        
        x_start = int(width * 0.095)
        x_end = width
        col_w = (x_end - x_start) / 24
        
        current_start = None
        for i in range(24):
            x_center = int(x_start + (i * col_w) + (col_w / 2))
            cv2.circle(debug_img, (x_center, y_center), 2, (0, 0, 255), -1)
            
            if y_center < height and x_center < width:
                px = img[y_center, x_center]
                is_blue = (LOWER_BLUE[0] <= px[0] <= UPPER_BLUE[0]) and \
                          (LOWER_BLUE[1] <= px[1] <= UPPER_BLUE[1]) and \
                          (LOWER_BLUE[2] <= px[2] <= UPPER_BLUE[2])
                
                time_val = hour_offset + (i * 0.5)
                if is_blue:
                    if current_start is None: current_start = time_val
                else:
                    if current_start is not None:
                        outage_intervals.append((current_start, time_val))
                        current_start = None
        if current_start is not None: outage_intervals.append((current_start, hour_offset + 12))

    scan_block(top_y_start, top_y_end, 0)
    scan_block(bottom_y_start, bottom_y_end, 12)
    return outage_intervals, debug_img

def format_intervals(intervals):
    if not intervals: return "✅ Світло є (або графік білий)."
    text = ""
    for start, end in intervals:
        s_h, s_m = int(start), int((start - int(start)) * 60)
        e_h, e_m = int(end), int((end - int(end)) * 60)
        end_str = f"{e_h:02}:{e_m:02}" if e_h != 24 else "24:00"
        text += f"`{s_h:02}:{s_m:02} - {end_str}`\n"
    return text

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# --- ГОЛОВНА ЛОГІКА ---

async def main():
    if not BOT_TOKEN:
        print("❌ Немає токена")
        return

    bot = Bot(token=BOT_TOKEN)
    urls = await asyncio.to_thread(get_image_links_headless)
    
    if not urls:
        print("❌ Нічого не знайдено.")
        await bot.session.close()
        return

    history = load_state()
    something_sent = False

    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None: continue

            # 1. Дата
            sched_date = parse_date_only(img)
            if not sched_date: 
                print(f"⚠️ Дата не прочиталась: {url}")
                continue
            date_str = sched_date.strftime("%d.%m.%Y")

            # 2. 🔥 РОЗУМНЕ ХЕШУВАННЯ
            # Вирізаємо ТІЛЬКИ вашу чергу і рахуємо її хеш
            queue_pixels = extract_queue_pixels(img)
            current_hash = calculate_hash(queue_pixels)

            # 3. Перевірка змін
            last_saved_hash = history.get(date_str)

            if last_saved_hash == current_hash:
                print(f"💤 Графік 3.1 на {date_str} не змінився.")
                continue
            
            # Якщо хеш змінився, значить пікселі у вашому рядку змінилися!
            status_text = "🔄 **ЗМІНИ В ГРАФІКУ!**" if last_saved_hash else "⚡️ **Новий графік**"
            print(f"🔥 Зміни для черги 3.1 на {date_str}!")

            intervals, debug_img = await asyncio.to_thread(analyze_schedule_image, img)
            text_schedule = format_intervals(intervals)
            
            caption = (
                f"{status_text}\n"
                f"📅 Дата: **{date_str}**\n\n"
                f"{text_schedule}"
            )

            is_success, buffer = cv2.imencode(".png", debug_img)
            if is_success:
                io_buf = BytesIO(buffer)
                await bot.send_photo(
                    chat_id=GROUP_ID,
                    photo=BufferedInputFile(io_buf.getvalue(), filename="schedule.png"),
                    caption=caption,
                    parse_mode="Markdown"
                )
                
                history[date_str] = current_hash
                something_sent = True

        except Exception as e:
            print(f"Error: {e}")

    if something_sent: save_state(history)
    await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
