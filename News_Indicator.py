import os
import time
import requests
import asyncio
from datetime import datetime, time as dt_time, timedelta
from bs4 import BeautifulSoup
from transformers import pipeline
from huggingface_hub import login
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
import matplotlib.pyplot as plt
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed  # Thêm as_completed vào đây

# Đăng nhập Hugging Face để sử dụng mô hình tóm tắt
login("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")

# Tải mô hình tóm tắt
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Cấu hình Telegram Bot
TELEGRAM_BOT_TOKEN = "7053413496:AAFuwt1b8ff2FKxKyTYaUv91IemHqLMTc0g"
CHAT_ID = "6632338735"

# Ánh xạ tên chỉ số/hàng hóa sang tiếng Việt
major_translation = {
    "Crude Oil": "Dầu thô", "Brent": "Dầu Brent", "Natural gas": "Khí tự nhiên",
    "Gasoline": "Xăng", "Coal": "Than đá", "TTF Gas": "Khí TTF", "Uranium": "Urani",
    "Urals Oil": "Dầu Urals", "Copper": "Đồng", "Gold": "Vàng", "HRC Steel": "Thép HRC",
    "Iron Ore": "Quặng sắt", "Silver": "Bạc", "Steel": "Thép", "Wheat": "Lúa mì",
    "Rubber": "Cao su", "Coffee": "Cà phê", "Rice": "Gạo", "Sugar": "Đường", "Urea": "Phân ure"
}

# Hàm tải dữ liệu từ web
def fetch_web_data(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải dữ liệu từ {url}: {e}")
        return None

# Hàm làm sạch tên chỉ số/hàng hóa
def clean_major_name(major):
    return major.split("\n\n")[0].strip() if "\n\n" in major else major.strip()

# Hàm trích xuất và lọc dữ liệu từ bảng HTML
def extract_and_filter_data(table, selected_indices, column_indices):
    rows = table.find_all('tr')[1:]
    filtered_data = []
    for row in rows:
        cols = [col.text.strip() for col in row.find_all(['td', 'th'])]
        if not cols or len(cols) < max(column_indices) + 1:
            continue
        major = clean_major_name(cols[column_indices[0]])
        if major in selected_indices:
            translated_major = major_translation.get(major, major)
            filtered_row = [translated_major, cols[column_indices[1]], cols[column_indices[2]]]
            if len(column_indices) > 3:
                filtered_row.append(cols[column_indices[3]])
            filtered_data.append(filtered_row)
    return filtered_data

# Hàm định dạng giá trị số
def format_value(value):
    try:
        if "." in value:
            integer_part, decimal_part = value.split(".", 1)
            return f"{integer_part}.{decimal_part.zfill(2)}"
        return f"{value}.00"
    except (ValueError, AttributeError):
        return value

# Hàm thêm emoji dựa trên phần trăm thay đổi
def get_emoji(value):
    try:
        numeric_value = float(value.strip('%'))
        return "🟢" if numeric_value > 0 else "🔴" if numeric_value < 0 else "🟡"
    except (ValueError, AttributeError):
        return "  "

# Hàm vẽ biểu đồ thay đổi hàng tuần
def plot_weekly_change(data, title):
    majors = [row[0] for row in data]
    weekly_changes = [float(row[3].strip('%') if row[3] else "0") if len(row) > 3 else 0 for row in data]
    plt.figure(figsize=(12, 6))
    bars = plt.barh(majors, weekly_changes, color=['green' if p > 0 else 'red' for p in weekly_changes])
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel("Phần trăm thay đổi (%)", fontsize=12)
    plt.ylabel("Chỉ số/Hàng hóa", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for bar in bars:
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.1f}%', va='center', ha='left')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    chart_file = f"chart_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()
    return chart_file

# Hàm tạo bảng HTML
def create_html_table(data, title):
    titles = ["💻", "Major", "Price", "%"]
    max_major_width = max(len(row[0]) for row in data) if data else len(titles[1])
    max_price_width = max(len(format_value(row[1])) for row in data) if data else len(titles[2])
    max_percentage_width = max(len(row[2]) for row in data) if data else len(titles[3])
    html_table = f"<b>{title}</b>\n\n<pre><code>\n"
    html_table += f"| {titles[0]:^2} | {titles[1]:^{max_major_width}} | {titles[2]:^{max_price_width}} | {titles[3]:^{max_percentage_width}} |\n"
    for row in data:
        emoji = get_emoji(row[2]) if len(row) > 2 else "  "
        html_table += f"| {emoji:^2} | {row[0]:<{max_major_width}} | {format_value(row[1]):>{max_price_width}} | {row[2]:>{max_percentage_width}} |\n"
    html_table += "</code></pre>"
    return html_table

# Hàm lấy dữ liệu cổ phiếu
def fetch_stocks_data():
    url = "https://tradingeconomics.com/stocks"
    content = fetch_web_data(url)
    if not content:
        return "Không thể tải dữ liệu cổ phiếu.", None
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return "Không đủ bảng dữ liệu cổ phiếu.", None
    table_1 = tables[0]
    table_4 = tables[3]
    selected_indices_table_1 = ["US500", "US30", "US100", "JP225", "GB100", "DE40", "FR40"]
    selected_indices_table_4 = ["VN", "HNX"]
    filtered_table_1 = extract_and_filter_data(table_1, selected_indices_table_1, [1, 2, 4, 5])
    filtered_table_4 = extract_and_filter_data(table_4, selected_indices_table_4, [1, 2, 4, 5])
    combined_data = filtered_table_1 + filtered_table_4
    html_table = create_html_table(combined_data, "Chỉ số cổ phiếu thế giới")
    chart_file = plot_weekly_change(combined_data, "Biến động tuần cổ phiếu")
    return html_table, chart_file

# Hàm lấy dữ liệu hàng hóa
def fetch_commodities_data():
    url = "https://tradingeconomics.com/commodities"
    content = fetch_web_data(url)
    if not content:
        return "Không thể tải dữ liệu hàng hóa.", None
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return "Không đủ bảng dữ liệu hàng hóa.", None
    combined_data = []
    selected_indices = {
        "Bảng 1": ["Crude Oil", "Brent", "Natural gas", "Gasoline", "Coal", "TTF Gas", "Uranium", "Urals Oil"],
        "Bảng 2": ["Copper", "Gold", "HRC Steel", "Iron Ore", "Silver", "Steel"],
        "Bảng 3": ["Wheat", "Rubber", "Coffee", "Rice", "Sugar"],
        "Bảng 4": ["Urea"]
    }
    for i, table in enumerate(tables[:4]):
        selected = selected_indices[f"Bảng {i+1}"]
        combined_data += extract_and_filter_data(table, selected, [0, 1, 3, 4])
    html_table = create_html_table(combined_data, "Giá cả hàng hóa")
    chart_file = plot_weekly_change(combined_data, "Biến động tuần hàng hóa")
    return html_table, chart_file

# Hàm lấy dữ liệu tài chính
def fetch_financials_data():
    url = "https://tradingeconomics.com/currencies"
    content = fetch_web_data(url)
    if not content:
        return "Không thể tải dữ liệu tài chính.", None
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return "Không đủ bảng dữ liệu tài chính.", None
    table_1 = tables[0]
    table_4 = tables[3]
    selected_indices_table_1 = ["DXY", "EURUSD", "USDJPY"]
    selected_indices_table_4 = ["USDVND"]
    filtered_table_1 = extract_and_filter_data(table_1, selected_indices_table_1, [1, 2, 4, 5])
    filtered_table_4 = extract_and_filter_data(table_4, selected_indices_table_4, [1, 2, 4, 5])
    combined_data = filtered_table_1 + filtered_table_4
    html_table = create_html_table(combined_data, "Tỷ giá tiền tệ")
    chart_file = plot_weekly_change(combined_data, "Biến động tuần tài chính")
    return html_table, chart_file

# Hàm chuyển đổi thời gian tương đối thành thời gian tuyệt đối
def parse_relative_time(relative_time, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    try:
        if "just now" in relative_time.lower():
            return reference_date
        elif "minute" in relative_time.lower():
            minutes = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(minutes=minutes)
        elif "hour" in relative_time.lower():
            hours = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(hours=hours)
        elif "day" in relative_time.lower():
            days = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(days=days)
        elif "week" in relative_time.lower():
            weeks = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(weeks=weeks)
        elif "month" in relative_time.lower():
            months = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(days=months * 30)
        elif "year" in relative_time.lower():
            years = int(re.search(r'\d+', relative_time).group())
            return reference_date - timedelta(days=years * 365)
        else:
            print(f"Thời gian không hợp lệ: {relative_time}")
            return None
    except Exception as e:
        print(f"Lỗi khi phân tích thời gian '{relative_time}': {e}")
        return None

# Hàm thu thập bài viết trong một ngày cụ thể
# Hàm thu thập bài viết trong một ngày cụ thể
def fetch_articles(max_articles=10, specific_date=None):
    if specific_date is None:
        specific_date = datetime.now()
    else:
        if isinstance(specific_date, str):
            specific_date = datetime.strptime(specific_date, "%d/%m/%Y")
    
    start_of_day = specific_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=1)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    )

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Lỗi khi khởi tạo ChromeDriver: {e}")
        return []

    articles = []
    seen_titles = set()
    try:
        driver.get("https://tradingeconomics.com/stream")
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.list-group-item.te-stream-item"))
        )

        last_height = driver.execute_script("return document.body.scrollHeight")
        max_attempts = 25
        for attempt in range(max_attempts):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print(f"Không có nội dung mới sau lần cuộn {attempt + 1}, dừng lại.")
                break
            last_height = new_height

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            article_elements = soup.select("li.list-group-item.te-stream-item")

            print(f"Số lượng bài viết tìm thấy sau cuộn {attempt + 1}: {len(article_elements)}")

            for i, elem in enumerate(article_elements):
                time_elem = elem.find("small")
                if not time_elem:
                    print(f"Bài {i}: Thiếu thời gian (small)")
                    continue

                relative_time = time_elem.text.strip()
                if not relative_time:
                    print(f"Bài {i}: Thời gian rỗng")
                    continue

                # Tìm tất cả thẻ <a> trong phần tử này
                title_elem_a = elem.find("a")  # Đổi từ class="te-stream-title-2" thành tìm thẻ <a> bất kỳ
                link = None
                title = None

                if title_elem_a:
                    # Tìm thẻ <b> bên trong <a> để lấy tiêu đề
                    bold_elem = title_elem_a.find("b")
                    if bold_elem:
                        title = bold_elem.text.strip()
                        print(f"Bài {i}: Tiêu đề từ thẻ <b> trong <a>: {title}")
                    else:
                        title = title_elem_a.text.strip()
                        print(f"Bài {i}: Tiêu đề từ thẻ <a>: {title}")

                    raw_link = title_elem_a.get("href")
                    if raw_link:
                        link = normalize_url(raw_link)
                        print(f"Bài {i}: Link: {link}")
                    else:
                        print(f"Bài {i}: Không tìm thấy href trong thẻ <a>")
                else:
                    # Nếu không tìm thấy thẻ <a>, thử tìm thẻ <b> trực tiếp
                    bold_elem = elem.find("b")
                    if bold_elem:
                        title = bold_elem.text.strip()
                        print(f"Bài {i}: Tiêu đề từ thẻ <b> trực tiếp: {title}")
                    else:
                        print(f"Bài {i}: Không tìm thấy thẻ <a> hoặc <b>")

                if not title or title in seen_titles:
                    print(f"Bài {i}: Tiêu đề rỗng hoặc đã tồn tại: {title}")
                    continue
                seen_titles.add(title)

                absolute_time = parse_relative_time(relative_time, reference_date=datetime.now())
                if absolute_time is None:
                    print(f"Bài {i}: Thời gian không hợp lệ: {relative_time}")
                    continue
                print(f"Bài {i}: Thời gian tuyệt đối: {absolute_time}, khoảng: {start_of_day} - {end_of_day}")

                if start_of_day <= absolute_time <= end_of_day:
                    articles.append({
                        "title": title,
                        "link": link,  # Cho phép link là None
                        "time": absolute_time
                    })
                else:
                    print(f"Bài {i}: Ngoài khoảng thời gian ngày {specific_date.strftime('%d/%m/%Y')}")

            if len(articles) >= max_articles:
                print(f"Đã đạt số lượng tối đa {max_articles} bài viết, dừng thu thập.")
                break

        print(f"Số lượng bài viết hợp lệ trong ngày {specific_date.strftime('%d/%m/%Y')}: {len(articles)}")

        articles = sorted(articles, key=lambda x: x["time"], reverse=True)[:max_articles]
        for article in articles:
            article["time"] = article["time"].strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(f"Lỗi khi thu thập bài viết: {e}")
    finally:
        driver.quit()

    return articles
# Hàm chuẩn hóa URL
def normalize_url(link):
    base_url = "https://tradingeconomics.com"
    if not link or not isinstance(link, str):
        return None
    if not link.startswith("http"):
        return urljoin(base_url, link)
    return link

# Hàm tóm tắt bài viết với bộ nhớ đệm
@lru_cache(maxsize=50)
def analyze_article(link):
    if not link or not isinstance(link, str):
        return "Lỗi: URL không hợp lệ hoặc không tồn tại."
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("h2", id="description")
        if content:
            content_text = content.get_text(separator="\n").strip()
            if len(content_text) > 100:
                summary = summarizer(content_text, max_length=130, min_length=30, do_sample=False)
                return summary[0]["summary_text"]
            else:
                return "Nội dung bài viết quá ngắn để tóm tắt."
        else:
            return "Không tìm thấy nội dung bài viết."
    except Exception as e:
        return f"Lỗi khi phân tích bài viết: {str(e)}"

# Hàm hỗ trợ tóm tắt song song
def summarize_articles(articles):
    valid_articles = [article for article in articles if article.get("link")]  # Lọc các bài có link hợp lệ
    if not valid_articles:
        return {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article = {executor.submit(analyze_article, article["link"]): article for article in valid_articles}
        summaries = {}
        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                summary = future.result()
                summaries[article["title"]] = summary
            except Exception as e:
                summaries[article["title"]] = f"Lỗi khi tóm tắt: {str(e)}"
    return summaries

# Lệnh /start
async def start_command(update: Update, context: CallbackContext):
    welcome_message = (
        "Chào mừng bạn đến với bot Trading Economics!\n"
        "Bot này cung cấp tin tức và dữ liệu thị trường từ Trading Economics.\n\n"
        "Các lệnh khả dụng:\n"
        "- /start: Hiển thị tin nhắn chào mừng\n"
        "- /help: Xem hướng dẫn\n"
        "- /news [dd/mm/yyyy]: Lấy 10 bài tin tức mới nhất trong ngày chỉ định (mặc định là hôm nay)\n"
        "- /stocks: Lấy dữ liệu chỉ số cổ phiếu\n"
        "- /commodities: Lấy dữ liệu giá hàng hóa\n"
        "- /financials: Lấy dữ liệu tài chính\n"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)

# Lệnh /help
async def help_command(update: Update, context: CallbackContext):
    help_message = (
        "Hướng dẫn sử dụng bot:\n\n"
        "Các lệnh khả dụng:\n"
        "- /start: Hiển thị tin nhắn chào mừng\n"
        "- /help: Xem hướng dẫn này\n"
        "- /news [dd/mm/yyyy]: Lấy 10 bài tin tức mới nhất trong ngày chỉ định (mặc định là hôm nay). Ví dụ: /news 03/09/2025\n"
        "- /stocks: Lấy dữ liệu chỉ số cổ phiếu thế giới\n"
        "- /commodities: Lấy dữ liệu giá cả hàng hóa\n"
        "- /financials: Lấy dữ liệu tỷ giá tiền tệ\n\n"
        "Bot cũng gửi báo cáo thị trường tự động hàng ngày."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_message)

# Lệnh /news
async def news_command(update: Update, context: CallbackContext):
    if context.args:
        try:
            specific_date = datetime.strptime(context.args[0], "%d/%m/%Y")
        except ValueError:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Định dạng ngày không hợp lệ. Vui lòng nhập theo định dạng dd/mm/yyyy, ví dụ: 03/09/2025")
            return
    else:
        specific_date = datetime.now()

    date_str = specific_date.strftime("%d/%m/%Y")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⏳ Đang thu thập tin tức trong ngày {date_str}, vui lòng đợi...")
    articles = fetch_articles(max_articles=10, specific_date=specific_date)
    
    if not articles:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Không tìm thấy bài viết nào trong ngày {date_str} hoặc có lỗi khi tải. Vui lòng kiểm tra log để biết chi tiết!")
        return
    elif len(articles) < 10:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"⚠️ Chỉ tìm thấy {len(articles)} bài viết trong ngày {date_str} thay vì 10. Kiểm tra log để biết chi tiết.")

    await context.bot.send_message(chat_id=update.effective_chat.id, text="📝 Đang tóm tắt bài viết...")
    summaries = summarize_articles(articles)

    message = f"📝 Tin tức mới nhất trong ngày {date_str}:\n\n"
    for article in articles:
        title = article["title"]
        summary = summaries.get(title, "Không thể tóm tắt bài viết này.")
        link = article["link"]
        time = article["time"]
        message += f"<b>{title}</b>\n<i>{time}</i>\n{summary}\n<a href='{link}'>Đọc thêm</a>\n\n" if link else f"<b>{title}</b>\n<i>{time}</i>\n{summary}\n"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode='HTML')

# Lệnh /stocks
async def stocks_command(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Đang tải dữ liệu cổ phiếu...")
    table, chart = fetch_stocks_data()
    if not table.startswith("Không thể"):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table, parse_mode='HTML')
        if chart and os.path.exists(chart):
            with open(chart, 'rb') as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)
            os.remove(chart)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table)

# Lệnh /commodities
async def commodities_command(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Đang tải dữ liệu hàng hóa...")
    table, chart = fetch_commodities_data()
    if not table.startswith("Không thể"):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table, parse_mode='HTML')
        if chart and os.path.exists(chart):
            with open(chart, 'rb') as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)
            os.remove(chart)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table)

# Lệnh /financials
async def financials_command(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Đang tải dữ liệu tài chính...")
    table, chart = fetch_financials_data()
    if not table.startswith("Không thể"):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table, parse_mode='HTML')
        if chart and os.path.exists(chart):
            with open(chart, 'rb') as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)
            os.remove(chart)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=table)

# Hàm gửi báo cáo tự động hàng ngày
async def send_daily_report(context: CallbackContext):
    for fetch_func, name in [
        (fetch_stocks_data, "cổ phiếu"),
        (fetch_commodities_data, "hàng hóa"),
        (fetch_financials_data, "tài chính")
    ]:
        await context.bot.send_message(chat_id=CHAT_ID, text=f"Đang tải dữ liệu {name}...")
        table, chart = fetch_func()
        if not table.startswith("Không thể"):
            await context.bot.send_message(chat_id=CHAT_ID, text=table, parse_mode='HTML')
            if chart and os.path.exists(chart):
                with open(chart, 'rb') as f:
                    await context.bot.send_photo(chat_id=CHAT_ID, photo=f)
                os.remove(chart)
        else:
            await context.bot.send_message(chat_id=CHAT_ID, text=table)
        time.sleep(1)

# Hàm chính
def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("stocks", stocks_command))
    application.add_handler(CommandHandler("commodities", commodities_command))
    application.add_handler(CommandHandler("financials", financials_command))
    application.job_queue.run_daily(send_daily_report, time=dt_time(hour=8, minute=0))
    print("🤖 Bot đang chạy...")
    application.run_polling()

if __name__ == "__main__":
    main()