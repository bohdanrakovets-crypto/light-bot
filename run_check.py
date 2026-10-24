import asyncio
import json
import os
import re
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pytesseract
import requests
from aiogram import Bot
from aiogram.types import BufferedInputFile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# --- НАЛАШТУВАННЯ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")
SITE_URL = "https://voe-poweron.inneti.net/schedule_queues"

pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Межі синього кольору у форматі BGR
LOWER_BLUE = np.array([80, 60, 40])
UPPER_BLUE = np.array([255, 180, 120])

TARGET_QUEUE_INDEX = 4  # Черга 3.1
STATE_FILE = "state.json"
MAX_HISTORY_DAYS = 60


# --- ФУНКЦІЇ ---

def get_image_links_headless() -> list[str]:
    """Запускає Chrome (Headless) і шукає посилання на картинки графіків."""
    print("🚀 Selenium: Start...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # Сучасний Selenium сам керує драйвером без webdriver-manager
    driver = webdriver.Chrome(options=chrome_options)
    found_urls = set()

    try:
        driver.set_page_load_timeout(30)
        driver.get(SITE_URL)
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
        except Exception:
            print("⚠️ Час очікування завантаження тегів <img> вичерпано.")

        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            src = img.get_attribute("src")
            if src and (
                "GPV" in src
                or ("media" in src and any(ext in src.lower() for ext in (".png", ".jpg", ".jpeg")))
            ):
                found_urls.add(src)
    except Exception as e:
        print(f"❌ Selenium Error: {e}")
    finally:
        driver.quit()

    return sorted(found_urls)


def parse_date_only(img: np.ndarray):
    """Витягує дату з шапки графіка за допомогою OCR."""
    try:
        h, w, _ = img.shape
        header_crop = img[0 : int(h * 0.15), 0 : int(w * 0.55)]
        gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        try:
            text = pytesseract.image_to_string(gray, lang="ukr+eng")
        except pytesseract.TesseractError:
            text = pytesseract.image_to_string(gray)

        dm = re.findall(r"(\d{2})\.(\d{2})\.(\d{4})", text)
        if dm:
            day, month, year = dm[0]
            return datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y").date()
    except Exception as e:
        print(f"⚠️ Помилка OCR: {e}")
    return None


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Об'єднує суміжні або перекриті інтервали (наприклад, 11:00-12:00 та 12:00-13:00 -> 11:00-13:00).
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current_start, current_end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        # Допуск 1e-5 на випадок похибки float
        if current_start <= prev_end + 1e-5:
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def analyze_schedule_image(img: np.ndarray):
    """
    Аналізує графік і повертає склеєний список інтервалів та розмічену картинку.
    """
    height, width, _ = img.shape
    debug_img = img.copy()
    rows_total = 12

    top_y_start = int(height * 0.19)
    top_y_end = int(height * 0.51)
    bottom_y_start = int(height * 0.58)
    bottom_y_end = int(height * 0.90)
    raw_intervals = []

    def scan_block(y_start: int, y_end: int, hour_offset: int):
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

            if 1 <= y_center < height - 1 and 1 <= x_center < width - 1:
                # Беремо патч 3x3 пікселі замість одного, щоб уникнути артефактів стиснення
                patch = img[y_center - 1 : y_center + 2, x_center - 1 : x_center + 2]
                px = np.median(patch, axis=(0, 1))

                is_blue = (
                    (LOWER_BLUE[0] <= px[0] <= UPPER_BLUE[0])
                    and (LOWER_BLUE[1] <= px[1] <= UPPER_BLUE[1])
                    and (LOWER_BLUE[2] <= px[2] <= UPPER_BLUE[2])
                )

                time_val = hour_offset + (i * 0.5)
                if is_blue:
                    if current_start is None:
                        current_start = time_val
                else:
                    if current_start is not None:
                        raw_intervals.append((current_start, time_val))
                        current_start = None

        if current_start is not None:
            raw_intervals.append((current_start, hour_offset + 12.0))

    scan_block(top_y_start, top_y_end, 0)
    scan_block(bottom_y_start, bottom_y_end, 12)

    # Склеюємо інтервали на стику 12:00 (і будь-які інші сусідні)
    merged_intervals = merge_intervals(raw_intervals)
    return merged_intervals, debug_img


def _float_to_time_str(val: float, is_end: bool = False) -> str:
    total_minutes = int(round(val * 60))
    h, m = divmod(total_minutes, 60)
    if is_end and h == 24:
        return "24:00"
    return f"{h:02d}:{m:02d}"


def format_intervals_to_string(intervals: list[tuple[float, float]]) -> str:
    """Перетворює список інтервалів у рядок підпису для state.json."""
    if not intervals:
        return "CLEAR"
    res = [
        f"{_float_to_time_str(start)}-{_float_to_time_str(end, is_end=True)}"
        for start, end in intervals
    ]
    return "|".join(res)


def format_intervals_pretty(intervals: list[tuple[float, float]]) -> str:
    """Гарний текст для повідомлення в Телеграм."""
    if not intervals:
        return "✅ Світло є (весь день без відключень)."
    lines = [
        f"🔴 <code>{_float_to_time_str(start)} - {_float_to_time_str(end, is_end=True)}</code>"
        for start, end in intervals
    ]
    return "\n".join(lines)


def normalize_signature_str(sig: str) -> str:
    """
    Нормалізує старі записи у state.json (склеює '11:00-12:00|12:00-13:00' -> '11:00-13:00'),
    щоб уникнути хибних спрацювань після оновлення коду.
    """
    if not sig or sig == "CLEAR" or "|" not in sig:
        return sig
    try:
        parsed = []
        for part in sig.split("|"):
            s_str, e_str = part.split("-")
            sh, sm = map(int, s_str.split(":"))
            eh, em = map(int, e_str.split(":"))
            parsed.append((sh + sm / 60.0, eh + em / 60.0))
        return format_intervals_to_string(merge_intervals(parsed))
    except Exception:
        return sig


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: normalize_signature_str(v) for k, v in data.items()}
        except Exception as e:
            print(f"⚠️ Помилка читання {STATE_FILE}: {e}")
    return {}


def save_state(state: dict):
    """Сортує історію за датою, залишає останні MAX_HISTORY_DAYS записів і зберігає у файл."""
    def parse_key(k):
        try:
            return datetime.strptime(k, "%d.%m.%Y")
        except ValueError:
            return datetime.min

    sorted_keys = sorted(state.keys(), key=parse_key)
    trimmed_keys = sorted_keys[-MAX_HISTORY_DAYS:]
    cleaned_state = {k: state[k] for k in trimmed_keys}

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_state, f, indent=2, ensure_ascii=False)


# --- ГОЛОВНА ЛОГІКА ---

async def main():
    if not BOT_TOKEN or not GROUP_ID:
        print("❌ Не задано BOT_TOKEN або GROUP_ID")
        return

    bot = Bot(token=BOT_TOKEN)
    try:
        urls = await asyncio.to_thread(get_image_links_headless)
        if not urls:
            print("❌ Картинок не знайдено.")
            return

        history = load_state()
        state_changed = False

        with requests.Session() as session:
            session.headers.update(
                {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )

            for url in urls:
                try:
                    resp = await asyncio.to_thread(session.get, url, timeout=15)
                    resp.raise_for_status()

                    img_arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    sched_date = await asyncio.to_thread(parse_date_only, img)
                    if not sched_date:
                        print(f"⚠️ Дата не розпізнана: {url}")
                        continue
                    date_str = sched_date.strftime("%d.%m.%Y")

                    intervals, debug_img = await asyncio.to_thread(
                        analyze_schedule_image, img
                    )
                    current_signature = format_intervals_to_string(intervals)
                    last_saved_signature = history.get(date_str)

                    if last_saved_signature == current_signature:
                        print(f"💤 Графік на {date_str} без змін ({current_signature}).")
                        continue

                    if last_saved_signature:
                        print(
                            f"🔥 ЗМІНИ на {date_str}! Було: {last_saved_signature} -> Стало: {current_signature}"
                        )
                        status_text = "🔄 <b>ЗМІНИ В ГРАФІКУ!</b>"
                    else:
                        print(f"✅ Новий графік на {date_str}: {current_signature}")
                        status_text = "⚡️ <b>Новий графік</b>"

                    text_schedule = format_intervals_pretty(intervals)
                    caption = (
                        f"{status_text}\n"
                        f"📅 Дата: <b>{date_str}</b>\n\n"
                        f"{text_schedule}"
                    )

                    is_success, buffer = cv2.imencode(".png", debug_img)
                    if is_success:
                        io_buf = BytesIO(buffer)
                        await bot.send_photo(
                            chat_id=GROUP_ID,
                            photo=BufferedInputFile(
                                io_buf.getvalue(), filename="schedule.png"
                            ),
                            caption=caption,
                            parse_mode="HTML",
                        )
                        history[date_str] = current_signature
                        state_changed = True

                except Exception as e:
                    print(f"❌ Помилка обробки {url}: {e}")

        if state_changed:
            save_state(history)

    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
