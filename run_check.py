import asyncio
import numpy as np
import cv2
import pytesseract
import re
import requests
import json
import os
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

# На GitHub Actions tesseract вже встановлений в систему
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

LOWER_BLUE = np.array([80, 60, 40])
UPPER_BLUE = np.array([255, 180, 120])
TARGET_QUEUE_INDEX = 4
STATE_FILE = "state.json"

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
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//img[contains(@src, 'GPV')]")))
        except: pass
        
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            src = img.get_attribute("src")
            if src and "GPV" in src: found_urls.append(src)
    except Exception as e:
        print(f"Selenium Error: {e}")
    finally:
        driver.quit()
    return list(set(found_urls))

def parse_header_info(img):
    try:
        h, w, _ = img.shape
        header_crop = img[0:int(h*0.15), 0:int(w*0.50)]
        gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, lang='ukr+eng')
        
        result = {'schedule_date': None, 'updated_at': datetime.min}
        
        dm = re.findall(r'(\d{2})\.(\d{2})\.(\d{4})', text)
        if dm:
            try: result['schedule_date'] = datetime.strptime(f"{dm[0][0]}.{dm[0][1]}.{dm[0][2]}", "%d.%m.%Y").date()
            except: pass
            
        tm = re.search(r'(\d{2}:\d{2})\s+(\d{2}\.\d{2}\.\d{4})', text)
        if tm:
            try: result['updated_at'] = datetime.strptime(f"{tm.group(2)} {tm.group(1)}", "%d.%m.%Y %H:%M")
            except: pass
            
        return result
    except: return {'schedule_date': None, 'updated_at': datetime.min}

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
            resp = requests.get(url, timeout=10)
            img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None: continue

            info = parse_header_info(img)
            sched_date = info['schedule_date']
            updated_at = info['updated_at']

            if not sched_date: continue

            date_str = sched_date.strftime("%d.%m.%Y")
            update_time_str = updated_at.strftime("%H:%M") if updated_at != datetime.min else "00:00"

            last_sent_time = history.get(date_str)
            if last_sent_time and last_sent_time >= update_time_str:
                print(f"💤 {date_str} вже був.")
                continue

            intervals, debug_img = await asyncio.to_thread(analyze_schedule_image, img)
            text_schedule = format_intervals(intervals)
            
            caption = (
                f"⚡️ **Оновлення ГПВ (Черга 3.1)**\n"
                f"📅 Дата: **{date_str}**\n"
                f"🕒 Оновлено: {update_time_str}\n\n"
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
                history[date_str] = update_time_str
                something_sent = True

        except Exception as e:
            print(f"Error: {e}")

    if something_sent: save_state(history)
    await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
