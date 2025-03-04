import os
import sys
import logging
import signal
import time
import random
import traceback
import base64
import requests
import hashlib
import threading
import concurrent.futures
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from instagrapi import Client  # pip install instagrapi
from moviepy.editor import VideoFileClip  # pip install moviepy
from PIL import Image  # pip install Pillow
from openai import OpenAI

from bs4 import BeautifulSoup  # pip install bs4
from google import genai  # pip install -q -U google-genai
from pythonjsonlogger import jsonlogger  # pip install python-json-logger
import redis  # pip install redis

OPENAI_API_KEY = "sk-svcacct-SE_lMM9mJ7Dcoxy0VPOf5ePMeM8UFsFK7TsxpxhYfxIyrF1pi_yleVkRqgR3BHSZRT3BlbkFJ3ZtiNz5U4ZrFoMiTKO9V1DmtLER1jZ_mvsX7jiJhVMoSlr3WIjgWoUMi53fFZonkAA"

client = OpenAI(api_key=OPENAI_API_KEY)  # pip install openai
openai = OpenAI

# ====================================================
# Structured Logging Configuration (JSON)
# ====================================================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s %(name)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# ====================================================
# Global Settings and Globals
# ====================================================
avoid_accounts = set()  # Accounts to avoid
banned_keywords = {"porn", "naked", "onlyfans"}
video_time_used = 0  # in seconds processed in current 30-day period
last_reset_time = datetime.now()

# Redis persistent cache for Gemini responses (ensure Redis is running)
cache = redis.Redis(host='localhost', port=6379, db=0)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ====================================================
# Global Credential Loading
# ====================================================
IG_USERNAME = "timeless._.trends_"
IG_PASSWORD = "18765618205"
GEMINI_API_KEY = "AIzaSyBs1__dL7uqt1dhsTvi9AXBpRL8271eZ0E"


DOWNLOAD_DIR = "downloads"
PROCESSED_DIR = "processed"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ====================================================
# Helper Function: Compute File Hash
# ====================================================
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ====================================================
# Enhanced Moderation: SafeSearch Detection
# ====================================================
def moderate_content(file_path):
    """
    Uses the Google Cloud Vision API SafeSearch feature to evaluate content.
    Returns True if content is acceptable, False if not.
    """
    try:
        with open(file_path, "rb") as img_file:
            img_data = img_file.read()
        encoded_image = base64.b64encode(img_data).decode("utf-8")
        payload = {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "SAFE_SEARCH_DETECTION", "maxResults": 1}]
                }
            ]
        }
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GEMINI_API_KEY}"
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            safe_search = result.get("responses", [{}])[0].get("safeSearchAnnotation", {})
            # Consider content unacceptable if any of these ratings are VERY_LIKELY or LIKELY.
            for category in ["adult", "racy"]:
                if safe_search.get(category, "UNKNOWN") in ["VERY_LIKELY", "LIKELY"]:
                    logger.info(f"Content moderation flagged {category} as {safe_search.get(category)}")
                    return False
            return True
        else:
            logger.error("SafeSearch API error: status code " + str(response.status_code))
            return True  # Fallback: assume safe if API fails
    except Exception as e:
        logger.error("Error during content moderation: " + str(e))
        traceback.print_exc()
        return True

# ====================================================
# Content Collector Module
# ====================================================
class ContentCollector:
    def __init__(self):
        self.reddit_endpoints = [
            "https://www.reddit.com/r/memes/top/.json?limit=5&t=day",
            "https://www.reddit.com/r/memes/top/.json?limit=5&t=all"
        ]
        self.headers = {"User-agent": "Mozilla/5.0"}

    def fetch_reddit_memes(self):
        logger.info("Fetching memes from Reddit...")
        media_items = []
        try:
            for url in self.reddit_endpoints:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        post_data = post["data"]
                        if post_data.get("post_hint") in ["image", "link"]:
                            image_url = post_data.get("url")
                            filename = os.path.join(DOWNLOAD_DIR, f"{post_data.get('id')}.jpg")
                            if self.download_file(image_url, filename):
                                media_items.append({
                                    "filepath": filename,
                                    "username": post_data.get("author"),
                                    "source": "reddit"
                                })
                else:
                    logger.error("Failed to fetch Reddit memes, status code: " + str(response.status_code))
        except Exception as e:
            logger.error("Error fetching Reddit memes: " + str(e))
            traceback.print_exc()
        logger.info(f"Collected {len(media_items)} media items from Reddit.")
        return media_items

    def fetch_instagram_media(self):
        logger.info("Fetching top media from Instagram...")
        media_items = []
        ig_client = Client()
        try:
            ig_client.login(IG_USERNAME, IG_PASSWORD)
        except Exception as e:
            logger.error("Error logging into Instagram for content collection: " + str(e))
            traceback.print_exc()
            return media_items
        hashtag = "funny"
        try:
            medias = ig_client.hashtag_medias_top(hashtag, amount=2)
        except Exception as e:
            logger.error("Error fetching Instagram hashtag top media: " + str(e))
            traceback.print_exc()
            return media_items

        for media in medias:
            media_id = media.pk
            if media.media_type == 1:
                url = media.thumbnail_url
                ext = ".jpg"
            elif media.media_type == 2:
                url = media.video_url
                ext = ".mp4"
            elif media.media_type == 8:  # Album; take first media item if available
                if hasattr(media, 'carousel_media') and media.carousel_media:
                    url = media.carousel_media[0].thumbnail_url
                else:
                    url = media.thumbnail_url
                ext = ".jpg"
            else:
                continue
            filename = os.path.join(DOWNLOAD_DIR, f"ig_{media_id}{ext}")
            if self.download_file(url, filename):
                try:
                    username = media.user.username
                except Exception:
                    username = "unknown"
                media_items.append({
                    "filepath": filename,
                    "username": username,
                    "source": "instagram"
                })
        logger.info(f"Collected {len(media_items)} media items from Instagram.")
        return media_items

    def download_file(self, url, filename):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                logger.info(f"Downloaded file: {filename}")
                return True
            else:
                logger.error(f"Failed to download file from {url}, status: {response.status_code}")
                return False
        except Exception as e:
            logger.error("Download error: " + str(e))
            return False

    def collect_all(self):
        items = []
        items.extend(self.fetch_reddit_memes())
        items.extend(self.fetch_instagram_media())
        filtered = [item for item in items if item.get("username", "").lower() not in 
                    {acc.lower() for acc in avoid_accounts}]
        return filtered

# ====================================================
# File Processor Module
# ====================================================
class FileProcessor:
    def __init__(self):
        pass

    def process_file(self, filepath):
        try:
            ext = os.path.splitext(filepath)[1].lower()
            output_file = os.path.join(PROCESSED_DIR, os.path.basename(filepath))
            if ext in [".jpg", ".jpeg", ".png"]:
                self.process_image(filepath, output_file)
            elif ext in [".mp4", ".mov", ".avi"]:
                self.process_video(filepath, output_file)
            elif ext in [".gif"]:
                self.process_gif(filepath, output_file)
            else:
                logger.warning(f"Unsupported file type: {filepath}")
                return None
            return output_file
        except Exception as e:
            logger.error("Error processing file " + filepath + ": " + str(e))
            traceback.print_exc()
            return None

    def process_image(self, input_path, output_path):
        try:
            img = Image.open(input_path)
            max_width = 1080
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                # Use Resampling.LANCZOS for high-quality downsampling.
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(output_path)
            logger.info(f"Processed image saved to {output_path}")
        except Exception as e:
            logger.error("Image processing error: " + str(e))
            traceback.print_exc()

    def process_video(self, input_path, output_path):
        try:
            clip = VideoFileClip(input_path)
            if clip.w > 720:
                clip = clip.resize(width=720)
            clip.write_videofile(output_file := output_path, codec="libx264", audio_codec="aac")
            logger.info(f"Processed video saved to {output_file}")
        except Exception as e:
            logger.error("Video processing error: " + str(e))
            traceback.print_exc()

    def process_gif(self, input_path, output_path):
        try:
            clip = VideoFileClip(input_path)
            clip.write_videofile(output_path, codec="libx264", audio=False)
            logger.info(f"Processed GIF saved to {output_path}")
        except Exception as e:
            logger.error("GIF processing error: " + str(e))
            traceback.print_exc()

# ====================================================
# AI Enhancer Module (Caption Generation, Multimodal Analysis & Moderation)
# ====================================================
class AIEnhancer:
    def __init__(self):
        pass

    def analyze_video_with_gemini(self, file_path):
        """
        Uses the Google Gen AI SDK to upload the video via the File API,
        waits until it's ACTIVE, and then generates content using Gemini.
        Uses Redis to cache responses.
        """
        file_hash = compute_file_hash(file_path)
        cache_key = f"gemini:{file_hash}"
        cached = cache.get(cache_key)
        if cached:
            logger.info("Using cached Gemini description for video.")
            return cached.decode("utf-8")
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Uploading video to Gemini File API...")
            video_file = client.files.upload(file=file_path)
            while video_file.state.name == "PROCESSING":
                logger.info("Waiting for video to be processed by Gemini...")
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)
            if video_file.state.name == "FAILED":
                logger.error("Video processing failed in Gemini File API.")
                return "a viral video"
            prompt_text = "Describe the video content in a positive, engaging tone in English."
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[video_file, prompt_text]
            )
            description = response.text
            cache.setex(cache_key, timedelta(days=30), description)
            logger.info("Gemini analysis description: " + description)
            return description
        except Exception as e:
            logger.error("Error analyzing video with Gemini: " + str(e))
            traceback.print_exc()
            return "a viral video"

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            encoded_image = base64.b64encode(img_data).decode("utf-8")
            payload = {
                "requests": [
                    {
                        "image": {"content": encoded_image},
                        "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
                    }
                ]
            }
            url = f"https://gemini.googleapis.com/v1/vision:annotate?key={GEMINI_API_KEY}"
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                labels = result.get("responses", [{}])[0].get("labelAnnotations", [])
                description = ", ".join(label.get("description", "") for label in labels)
                logger.info("Gemini analysis description: " + description)
                return description
            else:
                logger.error("Gemini Vision API error: status code " + str(response.status_code))
                return "a viral video"
        except Exception as e:
            logger.error("Error in Gemini Vision API: " + str(e))
            traceback.print_exc()
            return "a viral video"

    def generate_caption(self, file_path, metadata):
        """
        For video files, uses Gemini for analysis then uses OpenAI's ChatCompletion API
        (with GPT-4) to generate a caption. For other media types, uses OpenAI directly.
        Ensures captions are in English and include credit and hashtags.
        Implements exponential backoff and falls back to Gemini if necessary.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".mov", ".avi"]:
            try:
                clip = VideoFileClip(file_path)
                duration = clip.duration
            except Exception as e:
                logger.error("Error getting video duration: " + str(e))
                duration = 0
            global video_time_used, last_reset_time
            if (datetime.now() - last_reset_time).days >= 30:
                video_time_used = 0
                last_reset_time = datetime.now()
            if video_time_used + duration > 3600:
                logger.info("Video processing limit exceeded for this 30-day period. Using default caption.")
                return "Trending video! Credit: @" + metadata.get("username", "unknown")
            if not moderate_content(file_path):
                logger.info("Content moderation failed. Adding account to avoid list.")
                avoid_accounts.add(metadata.get("username", "unknown"))
                return "Content removed due to policy."
            description = self.analyze_video_with_gemini(file_path)
            for keyword in banned_keywords:
                if keyword in description.lower():
                    logger.info("Video contains banned keyword: " + keyword)
                    avoid_accounts.add(metadata.get("username", "unknown"))
                    return "Content removed due to policy."
            video_time_used += duration
            prompt = (
                f"Based on the following video analysis: '{description}', "
                f"generate an engaging, funny caption in English for the video. "
                f"Include credit as 'Credit: @{metadata.get('username', 'unknown')}' "
                f"and add a few relevant hashtags. Ensure the caption is positive and encourages growth."
            )
        else:
            prompt = (
                f"Generate an engaging, funny caption in English for the media file named '{os.path.basename(file_path)}'. "
                f"Include credit as 'Credit: @{metadata.get('username', 'unknown')}' and add a few relevant hashtags. "
                f"The caption should be positive, encourage growth, and not mention the platform. "
                f"Ensure that it sounds natural and entertaining."
            )
        max_retries = 5
        retry_delay = 1  # seconds
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # or "gpt-4"
                    messages=[
                        {"role": "system", "content": "You are a social media caption generator. Generate engaging, funny, and positive captions in English. Always include credit and relevant hashtags."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=40,
                    temperature=0.7,
                )

                caption = response.choices[0].message['content'].strip()
                logger.info("Generated caption: " + caption)
                return caption
            except openai.error.RateLimitError as e:
                logger.error(f"OpenAI API rate limit error on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.info("Max retries reached. Falling back to Gemini for caption generation.")
                    try:
                        client = genai.Client(api_key=GEMINI_API_KEY)
                        response_gemini = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[prompt]
                        )
                        caption = response_gemini.text.strip()
                        logger.info("Generated caption via Gemini: " + caption)
                        return caption
                    except Exception as ex:
                        logger.error("Gemini fallback error: " + str(ex))
                        traceback.print_exc()
                        return "Check out this awesome content! Credit: @" + metadata.get("username", "unknown")
            except Exception as e:
                logger.error("OpenAI API error: " + str(e))
                traceback.print_exc()
                return "Check out this awesome content! Credit: @" + metadata.get("username", "unknown")



# ====================================================
# Scheduler & Reposter Module
# ====================================================
class SchedulerReposter:
    def __init__(self):
        self.ig_client = Client()
        self.logged_in = False

    def login_instagram(self):
        try:
            self.ig_client.login(IG_USERNAME, IG_PASSWORD)
            self.logged_in = True
            logger.info("Logged into Instagram successfully.")
        except Exception as e:
            logger.error("Instagram login error: " + str(e))
            traceback.print_exc()
            self.logged_in = False

    def repost_content(self, media_file, caption):
        try:
            if not self.logged_in:
                self.login_instagram()
            self.ig_client.photo_upload(media_file, caption=caption)
            logger.info(f"Reposted content: {media_file} with caption: {caption}")
        except Exception as e:
            logger.error("Reposting error: " + str(e))
            traceback.print_exc()

    def repost_video_as_reel(self, media_file, caption, cover_image=None):
        try:
            if not self.logged_in:
                self.login_instagram()
            self.ig_client.video_upload_to_reels(media_file, caption=caption, cover=cover_image)
            logger.info(f"Reposted reel: {media_file} with caption: {caption}")
        except Exception as e:
            logger.error("Reels reposting error: " + str(e))
            traceback.print_exc()


    def repost_album(self, media_files, caption):
        try:
            if not self.logged_in:
                self.login_instagram()
            self.ig_client.album_upload(media_files, caption=caption)
            logger.info(f"Reposted album: {media_files} with caption: {caption}")
        except Exception as e:
            logger.error("Album reposting error: " + str(e))
            traceback.print_exc()

# ====================================================
# Minimal Flask Dashboard for Monitoring & Controls
# ====================================================
app = Flask(__name__)

@app.route('/')
def dashboard():
    status = {
        "scheduled_jobs": len(scheduler.get_jobs()) if scheduler else 0,
        "avoid_accounts": list(avoid_accounts),
        "video_time_used_seconds": video_time_used,
        "last_reset_time": last_reset_time.isoformat()
    }
    return jsonify(status)

@app.route('/jobs')
def jobs():
    jobs_list = []
    if scheduler:
        for job in scheduler.get_jobs():
            jobs_list.append({
                "id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None
            })
    return jsonify(jobs_list)

@app.route('/pause', methods=['POST'])
def pause_scheduler():
    if scheduler:
        scheduler.pause()
        return jsonify({"status": "paused"})
    return jsonify({"status": "scheduler not running"}), 500

@app.route('/resume', methods=['POST'])
def resume_scheduler():
    if scheduler:
        scheduler.resume()
        return jsonify({"status": "resumed"})
    return jsonify({"status": "scheduler not running"}), 500

@app.route('/avoid', methods=['GET', 'POST', 'DELETE'])
def avoid_list():
    global avoid_accounts
    if request.method == 'GET':
        return jsonify(list(avoid_accounts))
    elif request.method == 'POST':
        account = request.json.get("account", "").strip()
        if account:
            avoid_accounts.add(account)
            return jsonify({"status": "added", "account": account})
        else:
            return jsonify({"status": "error", "message": "No account provided"}), 400
    elif request.method == 'DELETE':
        avoid_accounts.clear()
        return jsonify({"status": "cleared"})

def start_dashboard():
    app.run(host='0.0.0.0', port=8080)

# ====================================================
# Persistent Job Function (Module Level)
# ====================================================
def collect_and_schedule_job():
    global collector, processor, enhancer, reposter, scheduler
    logger.info("Starting content collection task...")
    items = collector.collect_all()

    processed_items = []
    # Use ThreadPoolExecutor for asynchronous processing of files.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_item = {executor.submit(processor.process_file, item["filepath"]): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            processed_file = future.result()
            if processed_file:
                new_item = item.copy()
                new_item["processed_file"] = processed_file
                processed_items.append(new_item)

    processed_images = []
    for item in processed_items:
        ext = os.path.splitext(item["processed_file"])[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            processed_images.append(item)
        elif ext in [".mp4", ".mov", ".avi"]:
            caption = enhancer.generate_caption(item["processed_file"], item)
            delay = random.randint(60, 300)
            run_time = datetime.now() + timedelta(seconds=delay)
            scheduler.add_job(
                reposter.repost_video_as_reel,
                'date',
                run_date=run_time,
                args=[item["processed_file"], caption],
                misfire_grace_time=300
            )
            logger.info(f"Scheduled reel repost of {item['processed_file']} at {run_time}")
        else:
            caption = enhancer.generate_caption(item["processed_file"], item)
            delay = random.randint(60, 300)
            run_time = datetime.now() + timedelta(seconds=delay)
            scheduler.add_job(
                reposter.repost_content,
                'date',
                run_date=run_time,
                args=[item["processed_file"], caption],
                misfire_grace_time=300
            )
            logger.info(f"Scheduled repost of {item['processed_file']} at {run_time}")

    if processed_images:
        if len(processed_images) == 1:
            item = processed_images[0]
            caption = enhancer.generate_caption(item["processed_file"], item)
            delay = random.randint(60, 300)
            run_time = datetime.now() + timedelta(seconds=delay)
            scheduler.add_job(
                reposter.repost_content,
                'date',
                run_date=run_time,
                args=[item["processed_file"], caption],
                misfire_grace_time=300
            )
            logger.info(f"Scheduled repost of single image {item['processed_file']} at {run_time}")
        else:
            album_files = [item["processed_file"] for item in processed_images]
            caption = enhancer.generate_caption(processed_images[0]["processed_file"], processed_images[0])
            delay = random.randint(60, 300)
            run_time = datetime.now() + timedelta(seconds=delay)
            scheduler.add_job(
                reposter.repost_album,
                'date',
                run_date=run_time,
                args=[album_files, caption],
                misfire_grace_time=300
            )
            logger.info(f"Scheduled album repost of {album_files} at {run_time}")

# ====================================================
# Graceful Shutdown Handler
# ====================================================
def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Exiting gracefully...")
    sys.exit(0)

# ====================================================
# Main Automation Loop with Persistent Job Store
# ====================================================
def main():
    global collector, processor, enhancer, reposter, scheduler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    collector = ContentCollector()
    processor = FileProcessor()
    enhancer = AIEnhancer()
    reposter = SchedulerReposter()
    reposter.login_instagram()

    # Start the Flask dashboard in a background thread.
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()

    jobstores = {
        'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
    }
    scheduler = BackgroundScheduler(jobstores=jobstores)

    try:
        scheduler.remove_job('content_collection')
        logger.info("Removed existing 'content_collection' job.")
    except Exception as e:
        logger.info("No existing 'content_collection' job found.")

    scheduler.add_job(
        collect_and_schedule_job,
        'interval',
        hours=1,
        id='content_collection',
        next_run_time=datetime.now(),
        replace_existing=True
    )
    scheduler.start()
    logger.info("Automation system started with persistent job store. Running indefinitely...")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shutdown. Exiting program.")

if __name__ == "__main__":
    main()
