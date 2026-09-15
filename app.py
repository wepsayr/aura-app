import os
import uuid
import wave
import json
import random
import secrets
import requests
from datetime import datetime, date, timedelta
from collections import defaultdict
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g, flash
import numpy as np
import cv2
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
app.permanent_session_lifetime = timedelta(days=365)

DATABASE_URL = os.environ.get('DATABASE_URL')

DATA_DIR = os.environ.get('DATA_DIR', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Работа с БД ----------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = psycopg2.connect(DATABASE_URL)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query(sql, args=(), one=False):
    db = get_db()
    cur = db.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(sql, args)
        result = cur.fetchall()
        return (result[0] if result else None) if one else result
    finally:
        cur.close()

def execute(sql, args=(), returning=False):
    db = get_db()
    cur = db.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(sql, args)
        db.commit()
        if returning and cur.description is not None:
            return cur.fetchone()
        return None
    finally:
        cur.close()

def init_db():
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                email TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                activity TEXT,
                goal TEXT,
                theme TEXT DEFAULT 'blue',
                install_banner_closed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS habits (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                habit_name TEXT,
                category TEXT,
                target REAL,
                unit TEXT,
                frequency TEXT DEFAULT 'daily',
                active INTEGER DEFAULT 1,
                description TEXT,
                schedule_days TEXT DEFAULT '1,2,3,4,5,6,7'
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS habit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                habit_id INTEGER REFERENCES habits(id),
                date DATE,
                amount REAL,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                xp INTEGER DEFAULT 0,
                level INTEGER DEFAULT 1,
                current_streak INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                last_active_date DATE
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS voice_analyses (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                date DATE,
                duration REAL,
                rms REAL,
                pauses_count INTEGER,
                advice TEXT,
                exercises TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS style_analyses (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                date DATE,
                face_shape TEXT,
                skin_tone TEXT,
                dominant_colors TEXT,
                advice TEXT,
                exercises TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS xp_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                habit_id INTEGER,
                date DATE
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS bad_habits (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                habit_name TEXT,
                current_streak INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                last_active_date DATE,
                active INTEGER DEFAULT 1
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS bad_habit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                habit_id INTEGER REFERENCES bad_habits(id),
                date DATE,
                is_relapse INTEGER DEFAULT 0
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS posture_analyses (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                date DATE,
                front_shoulder_tilt REAL,
                front_neck_tilt REAL,
                side_neck_tilt REAL,
                back_shoulder_tilt REAL,
                advice TEXT,
                exercises TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                level_required INTEGER DEFAULT 1,
                gender_target TEXT DEFAULT 'all',
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                token TEXT UNIQUE,
                expires_at TIMESTAMP
            )
        ''')
        db.commit()
        cur.close()

        count = query('SELECT COUNT(*) AS c FROM articles', one=True)['c']
        if count == 0:
            articles_data = [
                ('Как голос влияет на первое впечатление',
                 'Твой голос — это твоя визитная карточка. Исследования показывают, что 38% первого впечатления о человеке формируется благодаря голосу. Уверенный, спокойный тембр вызывает доверие, а быстрая неразборчивая речь может создать впечатление неуверенности. Чтобы улучшить голос, начни с простого: записывай себя на диктофон и анализируй.',
                 'Голос', 'Тембр', 1, 'all'),
                ('Дыхание и голос: диафрагмальное дыхание',
                 'Диафрагмальное дыхание — основа сильного голоса. Попробуй упражнение «Книга на животе»: ляг на спину, положи книгу на живот. Вдыхай медленно через нос, чтобы книга поднималась. Выдыхай через рот, книга опускается. Повторяй по 5 минут в день.',
                 'Голос', 'Дыхание', 2, 'all'),
                ('Техника речи: как перестать использовать слова-паразиты',
                 'Слова-паразиты («ну», «типа», «как бы») выдают неуверенность. Записывай свою речь на диктофон и считай количество паразитов. Постепенно заменяй их паузами — это сделает речь весомее.',
                 'Голос', 'Дикция', 3, 'all'),
                ('Продвинутые техники дыхания для ораторов',
                 'Техника «4-7-8»: вдох на 4 счёта, задержка на 7, выдох на 8. Повтори 4 раза перед выступлением. Это успокаивает нервную систему и делает голос ровным.',
                 'Голос', 'Дыхание', 5, 'all'),
                ('Осанка и уверенность: научный подход',
                 'Ровная спина не только полезна для здоровья, но и напрямую влияет на уровень тестостерона и кортизола. Исследования гарвардского психолога Эми Кадди показали, что «поза силы» в течение 2 минут повышает уверенность на 20%. Начни с простого упражнения: каждые 30 минут вставай и тянись макушкой вверх.',
                 'Осанка', 'Уверенность', 1, 'all'),
                ('Сутулость и её последствия для здоровья',
                 'Сутулость не только портит внешний вид, но и сжимает внутренние органы, ухудшая пищеварение и дыхание. Чтобы исправить осанку, укрепи мышцы спины: упражнение «Лодочка» — лёжа на животе, одновременно поднимай руки и ноги, задерживай на 10 секунд.',
                 'Осанка', 'Сутулость', 2, 'all'),
                ('Растяжка для спины: 5 упражнений на каждый день',
                 '1. Наклоны в стороны стоя. 2. Кошка-корова на четвереньках. 3. Скручивания сидя. 4. Тяга коленей к груди лёжа. 5. Растяжка «Замок за спиной». Каждое упражнение выполняй по 30 секунд.',
                 'Осанка', 'Растяжка', 4, 'all'),
                ('Йога для осанки: комплекс для начинающих',
                 'Поза горы (Тадасана), поза дерева (Врикшасана), поза кобры (Бхуджангасана). Выполняй каждое утро в течение 10 минут. Улучшение осанки заметно уже через 2 недели.',
                 'Осанка', 'Йога', 6, 'all'),
                ('Цветотип: как подобрать одежду под свой тон кожи',
                 'Тёплый или холодный? Определи свой цветотип по венам на запястье: если они зеленоватые — у тебя тёплый подтон, если синие — холодный. Тёплым идут золотистые, персиковые, оливковые оттенки. Холодным — серебристые, голубые, изумрудные. Экспериментируй и находи свои идеальные сочетания!',
                 'Стиль', 'Цветотип', 1, 'all'),
                ('Секреты стиля: аксессуары, которые меняют образ',
                 'Правильно подобранные аксессуары могут полностью преобразить даже самый простой наряд. Часы, ремень, сумка, платок — выбирай что-то одно, но яркое. Не бойся экспериментировать с фактурами и цветами.',
                 'Стиль', 'Аксессуары', 3, 'all'),
                ('Мужской стиль: базовый гардероб',
                 'Базовый гардероб мужчины: 2 пары джинсов (тёмные и светлые), 3 однотонные футболки, 2 рубашки (белая и голубая), пиджак, классические кроссовки и туфли. Эти вещи сочетаются между собой и подходят для большинства случаев.',
                 'Стиль', 'Гардероб', 4, 'male'),
                ('Женский стиль: базовый гардероб',
                 'Базовый гардероб женщины: маленькое чёрное платье, юбка-карандаш, белая блузка, джинсы, удобные туфли-лодочки, жакет. Эти вещи легко комбинировать и они уместны в большинстве ситуаций.',
                 'Стиль', 'Гардероб', 4, 'female'),
                ('Создание собственного стиля: советы экспертов',
                 'Найди икону стиля, которая тебе нравится, и проанализируй, что именно привлекает. Адаптируй эти элементы под себя. Не бойся ошибаться — стиль вырабатывается с опытом.',
                 'Стиль', 'Гардероб', 6, 'all'),
                ('Как избавиться от привычки грызть ногти',
                 'Привычка грызть ногти часто связана с тревожностью. Попробуй заменить её на другое действие: носи с собой маникюрный набор, увлажняющий крем или эспандер. Каждый раз, когда хочется погрызть ноги, делай 5 глубоких вдохов.',
                 'Вредные привычки', 'Грызть ногти', 2, 'all'),
                ('Как отказаться от алкоголя: первые шаги',
                 'Отказ от алкоголя начинается с осознания триггеров. Замени вечерний бокал вина на травяной чай. Найди поддержку среди друзей или в сообществах. Помни: каждый день без алкоголя делает твою кожу чище, а сон крепче.',
                 'Вредные привычки', 'Алкоголь', 3, 'all'),
                ('Как бросить курить: стратегия маленьких шагов',
                 'Бросай курить постепенно: уменьшай количество сигарет на 1 каждые 2 дня. Замени привычку держать сигарету на зубочистку или леденец. Используй дыхательные упражнения при тяге. Помни: через 20 минут после последней сигареты давление приходит в норму.',
                 'Вредные привычки', 'Курение', 5, 'all')
            ]
            for title, content, category, subcategory, lvl, gender in articles_data:
                execute('INSERT INTO articles (title, content, category, subcategory, level_required, gender_target) VALUES (%s,%s,%s,%s,%s,%s)',
                        (title, content, category, subcategory, lvl, gender))

@app.before_request
def before_request():
    init_db()
    session.permanent = True

# ---------- Отправка почты через Brevo API ----------
def send_reset_email(to_email, reset_link):
    """Отправляет письмо со ссылкой для сброса пароля через Brevo HTTP API."""
    BREVO_API_KEY = os.environ.get('BREVO_API_KEY')
    BREVO_SENDER_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'aura.coach@yandex.ru')
    BREVO_SENDER_NAME = os.environ.get('BREVO_SENDER_NAME', 'Aura')

    if not BREVO_API_KEY:
        print("[MAIL] BREVO_API_KEY не задан", flush=True)
        return False

    html = f"""
    <html>
    <body style="font-family: Segoe UI, sans-serif; color: #1e3a5f;">
        <h2 style="color: #1565c0;">Сброс пароля в Aura</h2>
        <p>Привет!</p>
        <p>Ты запросил сброс пароля в приложении Aura.</p>
        <p>Перейди по ссылке ниже, чтобы задать новый пароль:</p>
        <p style="margin: 20px 0;">
            <a href="{reset_link}" style="background: #2196f3; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold;">
                Сбросить пароль
            </a>
        </p>
        <p style="color: #888; font-size: 0.9em;">Ссылка действительна 15 минут.</p>
        <p style="color: #888; font-size: 0.9em;">Если ты не запрашивал сброс, просто проигнорируй это письмо.</p>
        <p>С заботой, Aura 💎</p>
    </body>
    </html>
    """

    try:
        print(f"[MAIL] Отправляю письмо через Brevo на {to_email}...", flush=True)
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "accept": "application/json",
                "api-key": BREVO_API_KEY,
                "content-type": "application/json"
            },
            json={
                "sender": {
                    "name": BREVO_SENDER_NAME,
                    "email": BREVO_SENDER_EMAIL
                },
                "to": [{"email": to_email}],
                "subject": "Сброс пароля в Aura",
                "htmlContent": html
            },
            timeout=15
        )
        if response.status_code in (200, 201, 202):
            print("[MAIL] Письмо успешно отправлено через Brevo", flush=True)
            return True
        else:
            print(f"[MAIL] Ошибка Brevo: {response.status_code} — {response.text}", flush=True)
            return False
    except Exception as e:
        print(f"[MAIL] Ошибка отправки: {e}", flush=True)
        return False

# ---------- Вспомогательные функции ----------
def get_user():
    if 'user_id' in session:
        return query('SELECT * FROM users WHERE id = %s', (session['user_id'],), one=True)
    return None

def get_user_stats(user_id):
    stats = query('SELECT * FROM user_stats WHERE user_id = %s', (user_id,), one=True)
    if not stats:
        execute('INSERT INTO user_stats (user_id) VALUES (%s)', (user_id,))
        stats = query('SELECT * FROM user_stats WHERE user_id = %s', (user_id,), one=True)
    return stats

def create_default_habits(user_id):
    user = query('SELECT * FROM users WHERE id = %s', (user_id,), one=True)
    weight = user['weight'] or 70
    water_goal = round(weight * 0.033, 1)
    sleep_goal = 8 if user['age'] and user['age'] <= 25 else 7.5
    existing = query('SELECT habit_name FROM habits WHERE user_id = %s', (user_id,))
    existing_names = [row['habit_name'] for row in existing]
    defaults = [
        ('Вода', 'useful', water_goal, 'л', 'Пить достаточное количество воды для здоровья', '1,2,3,4,5,6,7'),
        ('Сон', 'useful', sleep_goal, 'ч', 'Полноценный сон для восстановления организма', '1,2,3,4,5,6,7'),
        ('Овощи', 'useful', 400, 'г', 'Ежедневное употребление овощей для витаминов', '1,2,3,4,5,6,7'),
        ('Прогулка', 'useful', 30, 'мин', 'Прогулка на свежем воздухе для активности', '1,2,3,4,5,6,7')
    ]
    for name, cat, target, unit, desc, sched in defaults:
        if name not in existing_names:
            execute('INSERT INTO habits (user_id, habit_name, category, target, unit, description, schedule_days) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                    (user_id, name, cat, target, unit, desc, sched))

def add_xp(user_id, amount):
    execute('UPDATE user_stats SET xp = xp + %s WHERE user_id = %s', (amount, user_id))
    stats = get_user_stats(user_id)
    new_level = stats['xp'] // 100 + 1
    if new_level > stats['level']:
        execute('UPDATE user_stats SET level = %s WHERE user_id = %s', (new_level, user_id))

def update_streak(user_id):
    stats = get_user_stats(user_id)
    today = date.today()
    last = stats['last_active_date']
    if last is None:
        new_streak = 1
    else:
        if isinstance(last, str):
            last_date = datetime.strptime(last[:10], '%Y-%m-%d').date()
        else:
            last_date = last
        if last_date == today:
            return
        elif last_date == today - timedelta(days=1):
            new_streak = stats['current_streak'] + 1
        else:
            new_streak = 1
    best = max(stats['best_streak'] or 0, new_streak)
    execute('UPDATE user_stats SET current_streak = %s, best_streak = %s, last_active_date = %s WHERE user_id = %s',
            (new_streak, best, today.isoformat(), user_id))
    if new_streak > 0 and new_streak % 7 == 0:
        add_xp(user_id, 50)

def get_today_habits(user_id):
    today = date.today()
    weekday = today.isoweekday()
    all_habits = query('SELECT * FROM habits WHERE user_id = %s AND active = 1', (user_id,))
    result = []
    for h in all_habits:
        sched = h['schedule_days'] or '1,2,3,4,5,6,7'
        days = [int(x.strip()) for x in sched.split(',') if x.strip().isdigit()]
        if weekday in days:
            log = query('SELECT amount FROM habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                        (user_id, h['id'], today.isoformat()), one=True)
            done = log['amount'] if log else 0
            row = dict(h)
            row['progress'] = done
            result.append(row)
    return result

def get_today_summary(user_id):
    habits = get_today_habits(user_id)
    total = len(habits)
    completed = sum(1 for h in habits if h['progress'] >= h['target'])
    return total, completed

def add_exercise_habits(user_id, exercises, schedule_days='1,2,3,4,5,6,7'):
    if not exercises:
        return
    for item in exercises:
        if isinstance(item, str):
            name = item
            desc = None
            target = 1
            unit = 'раз'
        else:
            name = item.get('name', '')
            desc = item.get('description', None)
            target = item.get('target', 1)
            unit = item.get('unit', 'раз')
            if 'schedule_days' in item:
                schedule_days = item['schedule_days']
        if not name or "попробуйте" in name.lower() or "фотографи" in name.lower():
            continue
        exists = query('SELECT id FROM habits WHERE user_id=%s AND habit_name=%s', (user_id, name), one=True)
        if not exists:
            execute('INSERT INTO habits (user_id, habit_name, category, target, unit, description, schedule_days) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                    (user_id, name, 'exercise', target, unit, desc, schedule_days))

def get_calendar_data(user_id):
    today = date.today()
    days = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        habits = query('SELECT * FROM habits WHERE user_id = %s AND active = 1', (user_id,))
        total = len(habits)
        completed = 0
        has_relapse = False
        for h in habits:
            log = query('SELECT amount FROM habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                        (user_id, h['id'], d.isoformat()), one=True)
            if log and log['amount'] >= h['target']:
                completed += 1
        bad_logs = query('SELECT is_relapse FROM bad_habit_log WHERE user_id=%s AND date=%s',
                         (user_id, d.isoformat()))
        if any(l['is_relapse'] == 1 for l in bad_logs):
            has_relapse = True
        if total == 0:
            color = 'gray'
        elif has_relapse:
            color = 'red'
        elif completed == total:
            color = 'green'
        elif completed > 0:
            color = 'yellow'
        else:
            color = 'gray'
        days.append({
            'date': d.strftime('%d.%m'),
            'weekday': ['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][d.weekday()],
            'color': color,
            'is_today': d == today
        })
    return days

def get_daily_tip(user):
    today = date.today()
    created = user['created_at']
    if created:
        if isinstance(created, str):
            created = datetime.strptime(created[:19], '%Y-%m-%d %H:%M:%S')
        if created.date() == today:
            return "Добро пожаловать в Aura! Я помогу тебе раскрыть твою внутреннюю и внешнюю силу."
    tips = [
        "Ты становишься лучше с каждым днём!",
        "Не забывай пить воду — это топливо для твоего голоса и кожи.",
        "Осанка — это каркас уверенности. Расправь плечи!",
        "Каждый день — новая возможность стать лучшей версией себя.",
        "Улыбнись! Ты прекрасно выглядишь.",
        "Твой голос — твоя визитная карточка. Удели ему минутку.",
        "Помни, что отдых так же важен, как и тренировки.",
        "Маленький шаг каждый день приводит к большим результатам.",
        "Будь добрее к себе. Ты уже делаешь великое дело.",
        "Сегодня отличный день, чтобы попробовать что-то новое!",
        "Твоя осанка определяет твою уверенность. Держи спину ровно!",
        "Голос — это инструмент. Настрой его на успех.",
        "Ты уникален. Твой стиль — твоё выражение.",
        "Забота о себе — это не эгоизм, а необходимость.",
        "Каждое выполненное упражнение делает тебя сильнее."
    ]
    bad = query('SELECT habit_name FROM bad_habits WHERE user_id=%s AND active=1', (user['id'],))
    if bad:
        bad_names = [row['habit_name'] for row in bad]
        if any('курение' in h.lower() for h in bad_names):
            tips.append("Без сигарет твой голос звучит чище. Держись!")
        if any('алкоголь' in h.lower() for h in bad_names):
            tips.append("Отказ от алкоголя улучшает кожу и сон. Ты справляешься!")
        if any('грызть ногти' in h.lower() for h in bad_names):
            tips.append("Твои руки достойны заботы. Оставь ногти в покое.")
    return random.choice(tips)

EXERCISES_MAP = {
    'курение': {'name': 'Дыхательная гимнастика при тяге к курению', 'description': '5–10 минут глубокого дыхания при появлении тяги', 'target': 1, 'unit': 'раз'},
    'алкоголь': {'name': 'Заменить алкоголь травяным чаем вечером', 'description': 'Выпить чашку успокаивающего чая (ромашка, мята) вместо алкоголя', 'target': 1, 'unit': 'раз'},
    'грызть ногти': {'name': 'Носить с собой маникюрный набор и увлажнять руки', 'description': 'При позыве грызть ногти – воспользуйся пилочкой или кремом', 'target': 1, 'unit': 'раз'},
    'переедание': {'name': 'Планировать меню на день и избегать триггеров', 'description': 'Составить список приёмов пищи и не отклоняться от него', 'target': 1, 'unit': 'раз'},
    'зависание в телефоне': {'name': 'Чтение книги перед сном вместо экрана', 'description': 'Минимум 20 минут чтения бумажной книги перед сном', 'target': 1, 'unit': 'раз'},
    'недосып': {'name': 'Ложиться спать на 30 минут раньше', 'description': 'Каждый день сдвигать время отхода ко сну на 30 минут раньше обычного', 'target': 1, 'unit': 'раз'},
    'кофеин': {'name': 'Заменить кофе на цикорий или воду с лимоном', 'description': 'Пить не более 1 чашки кофе в день, заменить остальные напитки', 'target': 1, 'unit': 'раз'},
    'срывы на близких': {'name': 'Дышать 10 секунд перед ответом', 'description': 'При возникновении раздражения сделать 3 глубоких вдоха и выдоха', 'target': 1, 'unit': 'раз'},
    'прокрастинация': {'name': 'Метод Pomodoro: 25 минут работы, 5 отдыха', 'description': 'Использовать таймер: 25 минут работы, 5 минут перерыва', 'target': 1, 'unit': 'раз'},
}
SUGGESTIONS = list(EXERCISES_MAP.keys())

def normalize_habit_name(name):
    return name.strip().capitalize()

def get_exercise_for_bad_habit(habit_name):
    name_lower = habit_name.lower()
    if name_lower in EXERCISES_MAP:
        return EXERCISES_MAP[name_lower]
    for key, val in EXERCISES_MAP.items():
        if key in name_lower or name_lower in key:
            return val
    return None

def add_exercise_for_bad_habit(habit_name, user_id):
    exercise_info = get_exercise_for_bad_habit(habit_name)
    if exercise_info:
        item = exercise_info
    else:
        item = {
            'name': f"Работа над устранением привычки «{habit_name}»: отслеживание и замена",
            'description': f"Упражнение для избавления от привычки «{habit_name}»",
            'target': 1,
            'unit': 'раз'
        }
    exists = query('SELECT id FROM habits WHERE user_id=%s AND habit_name=%s', (user_id, item['name']), one=True)
    if not exists:
        execute('INSERT INTO habits (user_id, habit_name, category, target, unit, description) VALUES (%s,%s,%s,%s,%s,%s)',
                (user_id, item['name'], 'exercise', item['target'], item['unit'], item['description']))
# ---------- Анализ голоса ----------
def analyze_audio(filepath):
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    if sampwidth == 1:
        dtype = np.uint8
        max_val = 255
    elif sampwidth == 2:
        dtype = np.int16
        max_val = 32767
    else:
        raise ValueError("Unsupported sample width")

    signal = np.frombuffer(audio_data, dtype=dtype)
    if n_channels == 2:
        signal = signal[::2]
    signal = signal.astype(np.float32) / max_val

    duration = len(signal) / framerate
    rms = np.sqrt(np.mean(signal**2))

    frame_len = int(framerate * 0.1)
    if frame_len < 1:
        frame_len = 512
    n_windows = len(signal) // frame_len
    if n_windows < 2:
        pauses = 0
    else:
        windowed = np.array_split(signal[:n_windows*frame_len], n_windows)
        rms_per_window = np.array([np.sqrt(np.mean(w**2)) for w in windowed])
        threshold = np.mean(rms_per_window) * 0.2
        silent = rms_per_window < threshold
        pauses = int(np.sum(np.diff(silent.astype(int)) == 1))

    advice = []
    exercises = []

    if rms < 0.02:
        advice.append("Твой голос звучит тихо. Поработай над громкостью и опорой дыхания.")
        exercises.append({'name': 'Пение гласных: тяни А, О, У, И, Э на разной громкости', 'description': '5 минут в день, постепенно увеличивая громкость', 'target': 1, 'unit': 'раз'})
    elif rms > 0.15:
        advice.append("Громкость высокая. Будь осторожен, чтобы не перенапрягать связки.")
        exercises.append({'name': "Техника 'шёпот → громко': начинай фразу шёпотом, плавно увеличивай громкость", 'description': '3 подхода по 5 повторений', 'target': 1, 'unit': 'раз'})
    else:
        advice.append("Хороший уровень громкости. Так держать!")

    if pauses > 25:
        advice.append("Слишком много пауз. Речь кажется прерывистой.")
        exercises.append({'name': "Записывай спонтанный монолог на 1 минуту без 'эээ'", 'description': '1–2 раза в день', 'target': 1, 'unit': 'раз'})
    elif pauses < 4 and rms > 0.01:
        advice.append("Речь почти без пауз. Добавь короткие паузы для выразительности.")
        exercises.append({'name': 'Читай текст, делая паузы после запятых и точек', 'description': '10 минут в день', 'target': 1, 'unit': 'раз'})

    if not exercises:
        exercises.append({'name': 'Ежедневно читай вслух 5–10 минут для самоконтроля', 'description': 'Читать вслух любимую книгу', 'target': 1, 'unit': 'раз'})

    if not advice:
        advice.append("Твой голос звучит отлично! Продолжай в том же духе.")

    return {
        'duration': round(float(duration), 2),
        'rms': round(float(rms), 4),
        'pauses': int(pauses),
        'advice': ' '.join(advice),
        'exercises': exercises
    }

# ---------- Анализ стиля ----------
def detect_face_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = None
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        pass
    if face_cascade is None:
        return None
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    aspect_ratio = w / h
    if aspect_ratio > 0.85:
        return "круглое"
    elif aspect_ratio < 0.75:
        return "вытянутое (овал/прямоугольник)"
    else:
        return "овальное"

def get_dominant_colors(image_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten())
    sorted_indices = np.argsort(counts)[::-1]
    colors = ['#%02x%02x%02x' % tuple(centers[i]) for i in sorted_indices]
    return colors

def analyze_style(filepath):
    if not os.path.exists(filepath):
        return {'face_shape': 'файл не найден', 'skin_tone': '', 'dominant_colors': '', 'advice': 'Не удалось найти загруженное фото. Попробуйте ещё раз.', 'exercises': []}

    img = cv2.imread(filepath)
    if img is None:
        return {'face_shape': 'не удалось определить', 'skin_tone': '', 'dominant_colors': '', 'advice': 'Не удалось открыть фото. Убедитесь, что загружаете правильный файл изображения (JPEG, PNG).', 'exercises': []}

    try:
        colors = get_dominant_colors(filepath, 3)
    except:
        colors = ["#2196f3", "#64b5f6", "#bbdefb"]

    rgb = colors[0].lstrip('#')
    r, g, b = int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16)
    brightness = (r + g + b) / 3
    skin_tone = "тёплый" if brightness > 128 else "холодный"

    face_shape = detect_face_shape(filepath)

    advice = []
    exercises = []

    if face_shape == "круглое":
        advice.append("У тебя круглое лицо. Рекомендуются причёски с объёмом на макушке и удлинённые стрижки.")
        exercises.append({'name': 'Попробуй укладку с косым пробором и асимметрией, чтобы визуально вытянуть лицо', 'description': 'Сделать укладку с боковым пробором', 'target': 1, 'unit': 'раз'})
    elif face_shape == "вытянутое (овал/прямоугольник)":
        advice.append("У тебя вытянутая форма лица. Подойдут чёлки, локоны и горизонтальные линии.")
        exercises.append({'name': 'Поэкспериментируй с пышной чёлкой и layered-стрижками', 'description': 'Посетить парикмахера для консультации', 'target': 1, 'unit': 'раз'})
    elif face_shape == "овальное":
        advice.append("Овальная форма лица считается универсальной. Тебе подойдут почти любые причёски!")
        exercises.append({'name': 'Попробуй асимметричные стрижки или собранные волосы, чтобы подчеркнуть скулы', 'description': 'Сделать высокий хвост или пучок', 'target': 1, 'unit': 'раз'})

    skin_advice = "Твой цветотип похож на {}. ".format(skin_tone)
    if skin_tone == "тёплый":
        skin_advice += "Тебе идут золотистые, персиковые, оливковые оттенки в одежде."
        exercises.append({'name': 'Добавь в гардероб вещи тёплых тонов: горчичный, коралловый, бежевый', 'description': 'Купить 1-2 вещи тёплых оттенков', 'target': 1, 'unit': 'раз'})
    else:
        skin_advice += "Тебе идут серебристые, синие, изумрудные, холодные розовые оттенки."
        exercises.append({'name': 'Добавь в гардероб вещи холодных тонов: голубой, сиреневый, мятный', 'description': 'Купить 1-2 вещи холодных оттенков', 'target': 1, 'unit': 'раз'})
    advice.append(skin_advice)

    if not exercises:
        exercises.append({'name': 'Носи то, что тебе нравится, и будь собой!', 'description': 'Получать удовольствие от своего стиля', 'target': 1, 'unit': 'раз'})

    return {
        'face_shape': face_shape if face_shape else "не удалось определить",
        'skin_tone': skin_tone,
        'dominant_colors': ', '.join(colors),
        'advice': ' '.join(advice),
        'exercises': exercises
    }

# ---------- Анализ осанки ----------
def analyze_front(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    body_gray = gray[y+h:img.shape[0], :]
    _, thresh = cv2.threshold(body_gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shoulder_tilt = 0.0
    if len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if len(contours) == 2:
            M1 = cv2.moments(contours[0])
            M2 = cv2.moments(contours[1])
            if M1['m00'] > 0 and M2['m00'] > 0:
                cx1 = int(M1['m10']/M1['m00'])
                cx2 = int(M2['m10']/M2['m00'])
                cy1 = int(M1['m01']/M1['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                shoulder_tilt = (cy2 - cy1) / (abs(cx2 - cx1) + 1e-5) * 10
    neck_tilt = (x + w/2 - img.shape[1]/2) / (img.shape[1]/2) * 10
    return {'shoulder_tilt': round(shoulder_tilt, 1), 'neck_tilt': round(neck_tilt, 1)}

def analyze_side(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    neck_tilt = (x + w - img.shape[1]/2) / (img.shape[1]/2) * 10
    return {'neck_tilt': round(neck_tilt, 1)}

def analyze_back(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    M1 = cv2.moments(contours[0])
    M2 = cv2.moments(contours[1])
    if M1['m00'] > 0 and M2['m00'] > 0:
        cx1 = int(M1['m10']/M1['m00'])
        cx2 = int(M2['m10']/M2['m00'])
        cy1 = int(M1['m01']/M1['m00'])
        cy2 = int(M2['m01']/M2['m00'])
        shoulder_tilt = (cy2 - cy1) / (abs(cx2 - cx1) + 1e-5) * 10
    else:
        shoulder_tilt = 0.0
    return {'shoulder_tilt': round(shoulder_tilt, 1)}

def analyze_posture(front_path, side_path, back_path):
    front_img = cv2.imread(front_path)
    side_img = cv2.imread(side_path)
    back_img = cv2.imread(back_path)

    front_data = analyze_front(front_img) if front_img is not None else None
    side_data = analyze_side(side_img) if side_img is not None else None
    back_data = analyze_back(back_img) if back_img is not None else None

    advice = []
    exercises = []

    if front_data:
        if abs(front_data['shoulder_tilt']) > 2:
            advice.append("Плечи несимметричны. Обрати внимание на равномерную нагрузку.")
            exercises.append({'name': "Тяга гантелей в наклоне", 'description': "3 подхода по 12 раз, 2-3 раза в неделю", 'target': 1, 'unit': 'раз', 'schedule_days': '1,3,5'})
        if abs(front_data['neck_tilt']) > 2:
            advice.append("Голова наклонена в сторону. Возможна привычка склонять голову при сидении.")
            exercises.append({'name': "Наклоны головы с удержанием", 'description': "По 10 наклонов в каждую сторону, 2 подхода ежедневно", 'target': 1, 'unit': 'раз'})
    if side_data:
        if abs(side_data['neck_tilt']) > 3:
            advice.append("Голова выдвинута вперёд (признак 'компьютерной шеи').")
            exercises.append({'name': "Упражнение 'Подбородок к шее'", 'description': "10 повторений, задерживая на 5 секунд, 2 раза в день", 'target': 1, 'unit': 'раз'})
    if back_data:
        if abs(back_data['shoulder_tilt']) > 2:
            advice.append("Со спины заметна асимметрия плеч. Проверь рабочее место.")
            exercises.append({'name': "Растяжка 'Замок за спиной'", 'description': "Держать 30 секунд, 3 подхода ежедневно", 'target': 1, 'unit': 'раз'})

    if not advice:
        advice.append("Осанка выглядит хорошо! Продолжай следить за собой.")
        exercises.append({'name': "Ежедневная разминка для спины и шеи", 'description': "5-10 минут лёгких упражнений каждый день", 'target': 1, 'unit': 'раз'})

    return {
        'front_shoulder_tilt': front_data['shoulder_tilt'] if front_data else 0.0,
        'front_neck_tilt': front_data['neck_tilt'] if front_data else 0.0,
        'side_neck_tilt': side_data['neck_tilt'] if side_data else 0.0,
        'back_shoulder_tilt': back_data['shoulder_tilt'] if back_data else 0.0,
        'advice': ' '.join(advice),
        'exercises': exercises
    }

# ---------- Маршруты ----------
@app.route('/')
def splash():
    user = get_user()
    return render_template('splash.html', user=user)

@app.route('/dashboard')
def dashboard():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    create_default_habits(user['id'])
    total, completed = get_today_summary(user['id'])
    stats = get_user_stats(user['id'])
    tip = get_daily_tip(user)
    calendar = get_calendar_data(user['id'])
    return render_template('index.html', user=user, stats=stats, total=total, completed=completed, tip=tip, calendar=calendar)

@app.route('/close_install_banner', methods=['POST'])
def close_install_banner():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    execute('UPDATE users SET install_banner_closed = 1 WHERE id = %s', (user['id'],))
    return jsonify({'success': True})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not username or not email or len(password) < 4:
            flash('Заполни все поля, пароль от 4 символов', 'error')
            return render_template('register.html')
        existing = query('SELECT id FROM users WHERE username = %s OR email = %s', (username, email), one=True)
        if existing:
            flash('Такой логин или email уже занят', 'error')
            return render_template('register.html')
        password_hash = generate_password_hash(password)
        new_user = execute('INSERT INTO users (username, email, password_hash) VALUES (%s,%s,%s) RETURNING id',
                           (username, email, password_hash), returning=True)
        session['user_id'] = new_user['id']
        return redirect(url_for('onboarding'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = query('SELECT * FROM users WHERE username = %s', (username,), one=True)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        else:
            flash('Неверный логин или пароль', 'error')
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = query('SELECT * FROM users WHERE email = %s', (email,), one=True)
        if user:
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=15)
            execute('INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (%s,%s,%s)',
                    (user['id'], token, expires_at.strftime('%Y-%m-%d %H:%M:%S')))
            reset_link = url_for('reset_password', token=token, _external=True)
            if send_reset_email(email, reset_link):
                flash('Ссылка для сброса отправлена на почту', 'success')
            else:
                flash('Не удалось отправить письмо. Проверь настройки почты.', 'error')
        else:
            flash('Если такой email зарегистрирован, письмо будет отправлено', 'success')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset = query('SELECT * FROM password_reset_tokens WHERE token = %s', (token,), one=True)
    if not reset:
        flash('Недействительная ссылка для сброса пароля', 'error')
        return redirect(url_for('forgot_password'))

    expires_at = reset['expires_at']
    if isinstance(expires_at, str):
        expires_at = datetime.strptime(expires_at[:19], '%Y-%m-%d %H:%M:%S')
    if datetime.utcnow() > expires_at:
        execute('DELETE FROM password_reset_tokens WHERE id = %s', (reset['id'],))
        flash('Срок действия ссылки истёк. Запроси сброс заново.', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('new_password', '')
        if len(new_password) < 4:
            flash('Пароль должен быть не короче 4 символов', 'error')
            return render_template('reset_password.html', token=token)
        new_hash = generate_password_hash(new_password)
        execute('UPDATE users SET password_hash = %s WHERE id = %s', (new_hash, reset['user_id']))
        execute('DELETE FROM password_reset_tokens WHERE id = %s', (reset['id'],))
        flash('Пароль успешно изменён! Теперь войди с новым паролем.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)

@app.route('/onboarding', methods=['GET', 'POST'])
def onboarding():
    user = get_user()
    if not user:
        return redirect(url_for('register'))
    if request.method == 'POST':
        data = request.form
        name = data.get('name', '').strip()
        gender = data.get('gender', '')
        age = data.get('age') or None
        height = data.get('height') or None
        weight = data.get('weight') or None
        activity = data.get('activity', '')
        goal = data.get('goal', '')
        execute('''UPDATE users SET name=%s, gender=%s, age=%s, height=%s, weight=%s, activity=%s, goal=%s WHERE id=%s''',
                (name, gender, age, height, weight, activity, goal, user['id']))
        return redirect(url_for('dashboard'))
    return render_template('onboarding.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/habits')
def habits():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    habits_list = get_today_habits(user['id'])
    stats = get_user_stats(user['id'])
    return render_template('habits.html', user=user, habits=habits_list, stats=stats)

@app.route('/log_habit', methods=['POST'])
def log_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    amount = float(data.get('amount', 0))
    if not habit_id:
        return jsonify({'error': 'No habit_id'}), 400
    today = date.today().isoformat()
    existing = query('SELECT id FROM habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                     (user['id'], habit_id, today), one=True)
    if existing:
        execute('UPDATE habit_log SET amount = %s WHERE id = %s', (amount, existing['id']))
    else:
        execute('INSERT INTO habit_log (user_id, habit_id, date, amount) VALUES (%s,%s,%s,%s)',
                (user['id'], habit_id, today, amount))

    habit = query('SELECT * FROM habits WHERE id = %s', (habit_id,), one=True)
    if habit and amount >= habit['target']:
        xp_entry = query('SELECT id FROM xp_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                         (user['id'], habit_id, today), one=True)
        if not xp_entry:
            add_xp(user['id'], 10)
            update_streak(user['id'])
            execute('INSERT INTO xp_log (user_id, habit_id, date) VALUES (%s,%s,%s)',
                    (user['id'], habit_id, today))

    habits_all = query('SELECT * FROM habits WHERE user_id = %s AND active = 1', (user['id'],))
    all_done = True
    for h in habits_all:
        log = query('SELECT amount FROM habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                    (user['id'], h['id'], today), one=True)
        if not log or log['amount'] < h['target']:
            all_done = False
            break
    if all_done:
        bonus_entry = query('SELECT id FROM xp_log WHERE user_id=%s AND habit_id=0 AND date=%s',
                            (user['id'], today), one=True)
        if not bonus_entry:
            add_xp(user['id'], 30)
            execute('INSERT INTO xp_log (user_id, habit_id, date) VALUES (%s,0,%s)',
                    (user['id'], today))
    return jsonify({'success': True})

@app.route('/reset_habit', methods=['POST'])
def reset_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    today = date.today().isoformat()
    execute('DELETE FROM habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
            (user['id'], habit_id, today))
    return jsonify({'success': True})

@app.route('/delete_habit', methods=['POST'])
def delete_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    if not habit_id:
        return jsonify({'error': 'No habit_id'}), 400
    execute('DELETE FROM habit_log WHERE habit_id=%s', (habit_id,))
    execute('DELETE FROM habits WHERE id=%s AND user_id=%s', (habit_id, user['id']))
    return jsonify({'success': True})

@app.route('/update_schedule', methods=['POST'])
def update_schedule():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    days = data.get('days', '')
    if not habit_id:
        return jsonify({'error': 'No habit_id'}), 400
    execute('UPDATE habits SET schedule_days = %s WHERE id = %s AND user_id = %s',
            (days, habit_id, user['id']))
    return jsonify({'success': True})

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age = request.form.get('age') or None
        gender = request.form.get('gender')
        height = request.form.get('height') or None
        weight = request.form.get('weight') or None
        activity = request.form.get('activity')
        goal = request.form.get('goal')
        theme = request.form.get('theme', 'blue')
        execute('''UPDATE users SET name=%s, age=%s, gender=%s, height=%s, weight=%s, activity=%s, goal=%s, theme=%s WHERE id=%s''',
                (name, age, gender, height, weight, activity, goal, theme, user['id']))
        return redirect(url_for('profile'))
    stats = get_user_stats(user['id'])
    return render_template('profile.html', user=user, stats=stats)

@app.route('/voice')
def voice():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    today = date.today().isoformat()
    last = query('SELECT * FROM voice_analyses WHERE user_id=%s AND date=%s ORDER BY id DESC LIMIT 1',
                 (user['id'], today), one=True)
    last_analysis = None
    if last:
        try:
            exercises = json.loads(last['exercises']) if last['exercises'] else []
        except:
            exercises = []
        last_analysis = {
            'duration': last['duration'],
            'rms': last['rms'],
            'pauses': last['pauses_count'],
            'advice': last['advice'],
            'exercises': exercises
        }
    return render_template('voice.html', user=user, last_analysis=last_analysis)

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)
    try:
        analysis = analyze_audio(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'Audio analysis failed: {str(e)}'}), 500
    execute('INSERT INTO voice_analyses (user_id, date, duration, rms, pauses_count, advice, exercises) VALUES (%s,%s,%s,%s,%s,%s,%s)',
            (user['id'], date.today().isoformat(), analysis['duration'], analysis['rms'],
             analysis['pauses'], analysis['advice'], json.dumps(analysis['exercises'])))
    add_exercise_habits(user['id'], analysis['exercises'])
    add_xp(user['id'], 15)
    update_streak(user['id'])
    os.remove(filepath)
    return jsonify(analysis)

@app.route('/style')
def style():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    today = date.today().isoformat()
    last = query('SELECT * FROM style_analyses WHERE user_id=%s AND date=%s ORDER BY id DESC LIMIT 1',
                 (user['id'], today), one=True)
    last_analysis = None
    if last:
        try:
            exercises = json.loads(last['exercises']) if last['exercises'] else []
        except:
            exercises = []
        last_analysis = {
            'face_shape': last['face_shape'],
            'skin_tone': last['skin_tone'],
            'dominant_colors': last['dominant_colors'],
            'advice': last['advice'],
            'exercises': exercises
        }
    return render_template('style.html', user=user, last_analysis=last_analysis)

@app.route('/analyze_style', methods=['POST'])
def analyze_style_route():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo file'}), 400
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    photo.save(filepath)
    try:
        analysis = analyze_style(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'Style analysis failed: {str(e)}'}), 500
    execute('''INSERT INTO style_analyses (user_id, date, face_shape, skin_tone, dominant_colors, advice, exercises)
               VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (user['id'], date.today().isoformat(), analysis['face_shape'], analysis['skin_tone'],
             analysis['dominant_colors'], analysis['advice'], json.dumps(analysis['exercises'])))
    add_exercise_habits(user['id'], analysis['exercises'])
    add_xp(user['id'], 15)
    update_streak(user['id'])
    os.remove(filepath)
    return jsonify(analysis)

@app.route('/posture')
def posture():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    today = date.today().isoformat()
    last = query('SELECT * FROM posture_analyses WHERE user_id=%s AND date=%s ORDER BY id DESC LIMIT 1',
                 (user['id'], today), one=True)
    last_analysis = None
    if last:
        try:
            exercises = json.loads(last['exercises']) if last['exercises'] else []
        except:
            exercises = []
        last_analysis = {
            'front_shoulder_tilt': last['front_shoulder_tilt'],
            'front_neck_tilt': last['front_neck_tilt'],
            'side_neck_tilt': last['side_neck_tilt'],
            'back_shoulder_tilt': last['back_shoulder_tilt'],
            'advice': last['advice'],
            'exercises': exercises
        }
    return render_template('posture.html', user=user, last_analysis=last_analysis)

@app.route('/analyze_posture', methods=['POST'])
def analyze_posture_route():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    if 'front' not in request.files or 'side' not in request.files or 'back' not in request.files:
        return jsonify({'error': 'Не все три фото загружены'}), 400
    front = request.files['front']
    side = request.files['side']
    back = request.files['back']
    if front.filename == '' or side.filename == '' or back.filename == '':
        return jsonify({'error': 'Одно из фото пустое'}), 400
    front_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_front.jpg")
    side_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_side.jpg")
    back_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_back.jpg")
    front.save(front_path)
    side.save(side_path)
    back.save(back_path)
    try:
        analysis = analyze_posture(front_path, side_path, back_path)
    except Exception as e:
        for p in [front_path, side_path, back_path]:
            if os.path.exists(p):
                os.remove(p)
        return jsonify({'error': f'Ошибка анализа: {str(e)}'}), 500
    execute('''INSERT INTO posture_analyses (user_id, date, front_shoulder_tilt, front_neck_tilt, side_neck_tilt, back_shoulder_tilt, advice, exercises)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
            (user['id'], date.today().isoformat(), analysis['front_shoulder_tilt'], analysis['front_neck_tilt'],
             analysis['side_neck_tilt'], analysis['back_shoulder_tilt'], analysis['advice'],
             json.dumps(analysis['exercises'])))
    add_exercise_habits(user['id'], analysis['exercises'])
    add_xp(user['id'], 15)
    update_streak(user['id'])
    for p in [front_path, side_path, back_path]:
        if os.path.exists(p):
            os.remove(p)
    return jsonify(analysis)

@app.route('/bad_habits')
def bad_habits():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    habits_list = query('SELECT * FROM bad_habits WHERE user_id=%s AND active=1', (user['id'],))
    return render_template('bad_habits.html', user=user, bad_habits=habits_list, suggestions=SUGGESTIONS)

@app.route('/add_bad_habit', methods=['POST'])
def add_bad_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Empty name'}), 400
    name = normalize_habit_name(name)
    exists = query('SELECT id FROM bad_habits WHERE user_id=%s AND habit_name=%s', (user['id'], name), one=True)
    if exists:
        return jsonify({'error': 'Уже есть такая привычка'}), 400
    execute('INSERT INTO bad_habits (user_id, habit_name) VALUES (%s,%s)', (user['id'], name))
    add_exercise_for_bad_habit(name, user['id'])
    return jsonify({'success': True})

@app.route('/log_bad_habit', methods=['POST'])
def log_bad_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    is_relapse = data.get('is_relapse', False)
    if not habit_id:
        return jsonify({'error': 'No habit_id'}), 400
    today = date.today()

    log = query('SELECT id, is_relapse FROM bad_habit_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                (user['id'], habit_id, today.isoformat()), one=True)
    if log:
        if not is_relapse and log['is_relapse'] == 1:
            execute('UPDATE bad_habit_log SET is_relapse=0 WHERE id=%s', (log['id'],))
        elif is_relapse and log['is_relapse'] == 0:
            execute('UPDATE bad_habit_log SET is_relapse=1 WHERE id=%s', (log['id'],))
    else:
        execute('INSERT INTO bad_habit_log (user_id, habit_id, date, is_relapse) VALUES (%s,%s,%s,%s)',
                (user['id'], habit_id, today.isoformat(), 1 if is_relapse else 0))

    habit = query('SELECT * FROM bad_habits WHERE id=%s', (habit_id,), one=True)
    if not habit:
        return jsonify({'error': 'Habit not found'}), 404
    logs = query('SELECT date, is_relapse FROM bad_habit_log WHERE habit_id=%s AND user_id=%s ORDER BY date ASC',
                 (habit_id, user['id']))
    current_streak = 0
    best_streak = habit['best_streak']
    if logs:
        def to_date(x):
            if isinstance(x, str):
                return datetime.strptime(x[:10], '%Y-%m-%d').date()
            return x
        sorted_logs = sorted(logs, key=lambda x: to_date(x['date']))
        streak = 0
        last_date = None
        for entry in reversed(sorted_logs):
            if entry['is_relapse'] == 1:
                break
            cur_date = to_date(entry['date'])
            if last_date is None:
                streak = 1
                last_date = cur_date
            elif (last_date - cur_date).days == 1:
                streak += 1
                last_date = cur_date
            else:
                break
        current_streak = streak
        max_streak = 0
        temp_streak = 0
        prev_date = None
        for entry in sorted_logs:
            if entry['is_relapse'] == 1:
                temp_streak = 0
                prev_date = None
            else:
                cur_date = to_date(entry['date'])
                if prev_date is None or (cur_date - prev_date).days == 1:
                    temp_streak += 1
                else:
                    temp_streak = 1
                prev_date = cur_date
                max_streak = max(max_streak, temp_streak)
        best_streak = max(habit['best_streak'] or 0, max_streak)

    execute('UPDATE bad_habits SET current_streak=%s, best_streak=%s, last_active_date=%s WHERE id=%s',
            (current_streak, best_streak, today.isoformat(), habit_id))

    if not is_relapse:
        xp_today = query('SELECT id FROM xp_log WHERE user_id=%s AND habit_id=%s AND date=%s',
                         (user['id'], -habit_id, today.isoformat()), one=True)
        if not xp_today:
            add_xp(user['id'], 5)
            execute('INSERT INTO xp_log (user_id, habit_id, date) VALUES (%s,%s,%s)',
                    (user['id'], -habit_id, today.isoformat()))

    return jsonify({'success': True, 'streak': current_streak, 'best': best_streak})

@app.route('/delete_bad_habit', methods=['POST'])
def delete_bad_habit():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json()
    habit_id = data.get('habit_id')
    if not habit_id:
        return jsonify({'error': 'No habit_id'}), 400
    execute('DELETE FROM bad_habit_log WHERE habit_id=%s', (habit_id,))
    execute('DELETE FROM bad_habits WHERE id=%s AND user_id=%s', (habit_id, user['id']))
    return jsonify({'success': True})

@app.route('/library')
def library():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    stats = get_user_stats(user['id'])
    user_level = stats['level']
    user_gender = user['gender'] if user['gender'] in ('male', 'female') else 'all'
    articles = query('SELECT * FROM articles WHERE gender_target IN (%s, \'all\') ORDER BY category, subcategory, level_required',
                     (user_gender,))
    library_data = {}
    for art in articles:
        cat = art['category'] if art['category'] else 'Общее'
        sub = art['subcategory'] if art['subcategory'] else 'Без подкатегории'
        if cat not in library_data:
            library_data[cat] = {}
        if sub not in library_data[cat]:
            library_data[cat][sub] = []
        library_data[cat][sub].append(art)
    for cat in library_data:
        for sub in library_data[cat]:
            library_data[cat][sub] = sorted(library_data[cat][sub],
                                            key=lambda x: (0 if x['level_required'] <= user_level else 1, x['level_required']))
    return render_template('library.html', user=user, library=library_data, user_level=user_level)

@app.route('/article/<int:article_id>')
def article(article_id):
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    art = query('SELECT * FROM articles WHERE id = %s', (article_id,), one=True)
    if not art:
        return "Статья не найдена", 404
    stats = get_user_stats(user['id'])
    if stats['level'] < art['level_required']:
        return "Недостаточно уровня для просмотра", 403
    if art['gender_target'] not in ('all', user['gender'] if user['gender'] in ('male', 'female') else 'all'):
        return "Статья недоступна для вашего пола", 403
    return render_template('article.html', user=user, article=art)

@app.route('/get_status')
def get_status():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 403
    stats = get_user_stats(user['id'])
    total, completed = get_today_summary(user['id'])
    return jsonify({
        'xp': stats['xp'],
        'level': stats['level'],
        'current_streak': stats['current_streak'],
        'best_streak': stats['best_streak'],
        'total_habits': total,
        'completed_habits': completed
    })

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
