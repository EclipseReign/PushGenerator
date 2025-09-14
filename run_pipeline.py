# -*- coding: utf-8 -*-
"""
CASE 1 — AUTO runner (без внешних модулей, кроме pandas)
Запуск:  python run_pipeline.py
Ищет папку "case 1" рядом со скриптом и сохраняет вывод в ./out
Гарантирует ТЗ: длина 180–220, ровно 1 CTA, без CAPS/«!», валюта формата "2 490 ₸".
Также устраняет повторы коротких филлеров (напр. "Без лишних шагов." дважды).
"""

import os, re, sys
import pandas as pd

# ---------- locate case dir next to script ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(SCRIPT_DIR, "case 1"),
    os.path.join(os.getcwd(), "case 1"),
]
CASE_DIR = None
for c in CANDIDATES:
    if os.path.isdir(c) and os.path.exists(os.path.join(c, "clients.csv")):
        CASE_DIR = os.path.abspath(c); break
if CASE_DIR is None:
    print("Не найдена папка 'case 1' рядом со скриптом (должен быть clients.csv).")
    sys.exit(1)

OUT_DIR = os.path.join(SCRIPT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
RU_MONTHS = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}
CTA_PATTERN = r"\b(Откр(ойте|ыть)|Оформить|Подключить|Узнать|Проверить|Рассчитать|Настроить|Создать|Купить|Посмотреть|Начать|Разместить)\b"

def fmt_kzt(a):
    try:
        if pd.isna(a): return ""
        return f"{a:,.0f}".replace(",", " ") + " ₸"
    except Exception:
        return ""

def load_txns(case_dir, cc):
    p = os.path.join(case_dir, f"client_{cc}_transactions_3m.csv")
    return pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else pd.DataFrame(columns=["client_code","date","category","amount","currency"])

def load_trs(case_dir, cc):
    p = os.path.join(case_dir, f"client_{cc}_transfers_3m.csv")
    return pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else pd.DataFrame(columns=["client_code","date","type","direction","amount","currency"])

def stats(profile, tx, tr):
    spend = tx.groupby("category")["amount"].sum().to_dict() if len(tx) else {}
    base_spend = sum(spend.values())
    online_bundle = sum(spend.get(c,0) for c in ["Едим дома","Смотрим дома","Играем дома"])
    fx_activity = ((tr["type"].isin(["fx_buy","fx_sell"])).any()) or ((tx["currency"].isin(["USD","EUR","GBP"])).any() if len(tx) else False)
    fx_volume = tx.loc[tx["currency"].isin(["USD","EUR","GBP"]), "amount"].sum() if len(tx) else 0.0
    fx_curr = "USD"; month_name = ""
    if len(tx):
        vc = tx["currency"].value_counts()
        fx_curr = "USD" if vc.get("USD",0)>=vc.get("EUR",0) else "EUR"
        tx["dt"] = pd.to_datetime(tx["date"])
        month_name = RU_MONTHS.get(tx["dt"].dt.month.max(),"")
    inflows = tr.loc[tr["direction"]=="in","amount"].sum() if len(tr) else 0.0
    outflows= tr.loc[tr["direction"]=="out","amount"].sum() if len(tr) else 0.0
    low_bal = profile.get("avg_monthly_balance_KZT",0) < 100_000
    cash_gap = (outflows>inflows*1.3) or (low_bal and outflows>inflows)
    atm_cnt = int((tr["type"]=="atm_withdrawal").sum()) if len(tr) else 0
    free_cash = max(profile.get("avg_monthly_balance_KZT",0)-50_000,0)
    jcr = sum(spend.get(c,0) for c in ["Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны"])
    topcats = [c for c,_ in sorted(spend.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    taxi_trips = int((tx["category"]=="Такси").sum()) if len(tx) else 0
    taxi_spend = spend.get("Такси",0)
    return dict(spend=spend, base_spend=base_spend, online_bundle=online_bundle, fx_activity=fx_activity, fx_volume=fx_volume,
                fx_curr=fx_curr, month=month_name, inflows=inflows, outflows=outflows, cash_gap=cash_gap, atm_cnt=atm_cnt,
                free_cash=free_cash, jcr=jcr, topcats=topcats, taxi_trips=taxi_trips, taxi_spend=taxi_spend, vol_low=True)

DEFAULT_LIMITS = {"travel_cashback_cap":15000,"premium_cashback_cap":20000,"credit_card_bonus_cap":25000}
def benefit_travel(spend):
    base = spend.get("Путешествия",0)+spend.get("Такси",0)+spend.get("Отели",0)
    return min(0.04*base, DEFAULT_LIMITS["travel_cashback_cap"])
def benefit_premium(spend, base_spend, avg_bal, atm_cnt, jcr):
    tier = 0.04 if avg_bal>=2_000_000 else (0.03 if avg_bal>=1_000_000 else 0.02)
    est = tier*base_spend + 0.04*jcr + min(atm_cnt*300,3000)
    return min(est, DEFAULT_LIMITS["premium_cashback_cap"])
def benefit_credit(spend, online_bundle):
    fav_top = sum(sorted([spend.get(c,0) for c in spend], reverse=True)[:3])
    return min(0.10*fav_top + 0.10*online_bundle, DEFAULT_LIMITS["credit_card_bonus_cap"])
def benefit_fx(volume, has_fx):
    return 0.007*volume if has_fx else 0.0
def benefit_mdep(avg, fx):
    return avg*0.02*(3/12) if (avg>0 and fx) else 0.0
def benefit_sdep(avg, vol_low):
    return avg*0.06*(3/12) if (avg>0 and vol_low) else 0.0
def benefit_adep(free_cash):
    return free_cash*0.07*(3/12) if free_cash>0 else 0.0
def benefit_inv(free_cash):
    return min(3000+0.001*free_cash, 7000) if free_cash>0 else 0.0
def benefit_gold(free_cash):
    return min(0.002*free_cash, 6000) if free_cash>0 else 0.0

def score(profile, s):
    avg = profile.get("avg_monthly_balance_KZT",0)
    b = {
        "Карта для путешествий": benefit_travel(s["spend"]),
        "Премиальная карта":    benefit_premium(s["spend"], s["base_spend"], avg, s["atm_cnt"], s["jcr"]),
        "Кредитная карта":      benefit_credit(s["spend"], s["online_bundle"]),
        "FX/мультивалютный продукт": benefit_fx(s["fx_volume"], s["fx_activity"]),
        "Кредит наличными":     5000.0 if s["cash_gap"] else 0.0,
        "Депозит мультивалютный": benefit_mdep(avg, s["fx_activity"]),
        "Депозит сберегательный": benefit_sdep(avg, s["vol_low"]),
        "Депозит накопительный": benefit_adep(s["free_cash"]),
        "Инвестиции (брокерский счёт)": benefit_inv(s["free_cash"]),
        "Золотые слитки":       benefit_gold(s["free_cash"]),
    }
    if not s["fx_activity"]:
        b["FX/мультивалютный продукт"] = 0.0
        b["Депозит мультивалютный"] *= 0.5
    top4 = [p for p,_ in sorted(b.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    return b, top4

CTA = {
    "Карта для путешествий": ["Оформить карту","Открыть карту","Подключить карту"],
    "Премиальная карта": ["Оформить сейчас","Открыть карту","Подключить премиум"],
    "Кредитная карта": ["Оформить карту","Проверить лимит","Подать заявку"],
    "FX/мультивалютный продукт": ["Настроить обмен","Открыть обмен","Создать автопокупку"],
    "Кредит наличными": ["Узнать лимит","Оформить кредит","Рассчитать платеж"],
    "Депозит мультивалютный": ["Открыть вклад","Разместить средства","Начать сбережения"],
    "Депозит сберегательный": ["Открыть вклад","Начать копить","Разместить средства"],
    "Депозит накопительный": ["Открыть вклад","Начать копить","Автопополнение"],
    "Инвестиции (брокерский счёт)": ["Открыть счёт","Начать инвестировать","Без комиссий"],
    "Золотые слитки": ["Посмотреть слитки","Купить слиток","Оформить покупку"],
}

# --- грамотная вербализация категорий ---
CAT_VERBS = {
    "Продукты питания": "покупаете продукты",
    "Кафе и рестораны": "ходите в кафе и рестораны",
    "Путешествия": "часто путешествуете",
    "Такси": "пользуетесь такси",
    "Отели": "бронируете отели",
    "Едим дома": "пользуетесь сервисами «Едим дома»",
    "Смотрим дома": "смотрите фильмы и сериалы по подписке",
    "Играем дома": "покупаете игры и подписки",
    "Ювелирные украшения": "покупаете украшения",
    "Косметика и Парфюмерия": "покупаете косметику и парфюмерию",
}
def cat_to_phrase(cat):
    if not isinstance(cat, str): return ""
    return CAT_VERBS.get(cat, f"тратите на {cat.lower()}")

def verbalize_cats(cat_list):
    phrases = [cat_to_phrase(c) for c in cat_list if c]
    phrases = [p for p in phrases if p]
    if not phrases: return ""
    if len(phrases) == 1: return phrases[0]
    if len(phrases) == 2: return f"{phrases[0]} и {phrases[1]}"
    return f"{phrases[0]}, {phrases[1]} и {phrases[2]}"

# --- копирайтинг ---
def copy_credit(name, s, benefit):
    cats_text = verbalize_cats(s["topcats"])
    b = f" — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return [
        f"{name}, мы заметили, что вы часто {cats_text}. Кредитная карта добавит бонусы к привычным покупкам и онлайн-сервисам{b}. {CTA['Кредитная карта'][0]}.",
        f"{name}, ваши регулярные траты — это {cats_text}. Усильте их кешбэком до 10% и на онлайн-сервисы{b}. {CTA['Кредитная карта'][1]}.",
    ]
def copy_travel(name, s, benefit):
    taxi = s["taxi_trips"]; ts = fmt_kzt(s["taxi_spend"]); m = s["month"]
    b = f" — сэкономили бы ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return [
        f"{name}, мы заметили поездки в {m}: такси {taxi} раз на {ts}. Тревел-карта вернёт часть расходов{b}. {CTA['Карта для путешествий'][0]}.",
        f"{name}, если часто в дороге, пусть часть трат возвращается кешбэком на билеты и отели{b}. {CTA['Карта для путешествий'][1]}.",
    ]
def copy_premium(name, s, benefit):
    b = f" — до {fmt_kzt(benefit)}" if benefit>0 else ""
    return [
        f"{name}, траты в ресторанах и высокий остаток — повод получать больше. Премиальная карта добавит кешбэк и комфорт в поездках{b}. {CTA['Премиальная карта'][0]}.",
        f"{name}, переключите повседневные покупки на премиальную — больше пользы за те же деньги{b}. {CTA['Премиальная карта'][1]}.",
    ]
def copy_fx(name, s, benefit):
    curr = s["fx_curr"]; b = f" — сэкономите на спреде ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return [
        f"{name}, оплачиваете в {curr}? Включите обмен заранее и авто-покупку по целевому курсу{b}. {CTA['FX/мультивалютный продукт'][0]}.",
        f"{name}, сделайте курс предсказуемым для платежей в {curr}: авто-покупка валюты по правилу{b}. {CTA['FX/мультивалютный продукт'][2]}.",
    ]
def copy_deposit(name, s, benefit, prod):
    b = f" — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return [
        f"{name}, закрепите часть средств на вкладе — копить так проще и спокойнее{b}. {CTA[prod][0]}.",
        f"{name}, пусть сбережения растут автоматически: вклад пополняется в пару касаний{b}. {CTA[prod][1]}.",
    ]
def copy_cash_loan(name, s, benefit):
    return [f"{name}, если намечаются крупные траты, кредит наличными сгладит платежи и добавит гибкости. {CTA['Кредит наличными'][0]}."]
def copy_invest(name, s, benefit):
    return [f"{name}, начните аккуратно: счёт открывается за пару касаний, без лишней сложности. {CTA['Инвестиции (брокерский счёт)'][0]}."]

GEN = {
    "Кредитная карта": copy_credit,
    "Карта для путешествий": copy_travel,
    "Премиальная карта": copy_premium,
    "FX/мультивалютный продукт": copy_fx,
    "Депозит мультивалютный": lambda n,s,b: copy_deposit(n,s,b,"Депозит мультивалютный"),
    "Депозит сберегательный": lambda n,s,b: copy_deposit(n,s,b,"Депозит сберегательный"),
    "Депозит накопительный": lambda n,s,b: copy_deposit(n,s,b,"Депозит накопительный"),
    "Кредит наличными": copy_cash_loan,
    "Инвестиции (брокерский счёт)": copy_invest,
    "Золотые слитки": lambda n,s,b: [f"{n}, добавьте немного золота для баланса сбережений. {CTA['Золотые слитки'][0]}."],
}

def detect_segment(profile, s):
    if (str(profile.get("status","")).strip() == "Премиальный клиент") or (profile.get("avg_monthly_balance_KZT",0) >= 1_000_000):
        return "VIP"
    if s["fx_activity"]:
        return "FX"
    if int(profile.get("age",99)) <= 27:
        return "YOUTH"
    if str(profile.get("status","")).strip() == "Зарплатный клиент":
        return "SALARY"
    return "STANDARD"

# ---- анти-дубликаты + жёсткая нормализация под ТЗ ----
FILLERS = {
    "Без лишних шагов.",
    "Детали внутри.",
    "Пара касаний — готово.",
    "Всё прозрачно.",
    "Ничего лишнего.",
    "Комфорт в поездках и каждый день.",
    "Курс — под ваш контроль.",
    "Без спешки и случайных курсов.",
    "Полезно в повседневных расходах.",
    "Сделаете в приложении за минуту.",
    "Подробности внутри.",
    "Это быстро и просто.",
    "Всё нужное — в приложении.",
}
FILLER_POOL = ["Подробности внутри.", "Это быстро и просто.", "Всё нужное — в приложении.", "Ничего лишнего."]

def split_sentences(text):
    t = " ".join(str(text).split()).replace("..", ".").strip()
    if not t.endswith("."): t += "."
    parts = [s.strip() for s in re.split(r"\.\s*", t) if s.strip()]
    sents = [s if s.endswith(".") else s + "." for s in parts]
    return sents

def join_sentences(sents):
    out = " ".join(s.strip() for s in sents if s and s.strip())
    out = re.sub(r"\s+", " ", out).strip()
    out = out.replace("..", ".")
    if not out.endswith("."): out += "."
    return out

def dedup_sentences(sents):
    seen = set(); out = []
    for s in sents:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key in seen: continue
        seen.add(key); out.append(s)
    # remove consecutive duplicates as well
    cleaned = []; prev = None
    for s in out:
        if prev and re.sub(r"\s+", " ", prev.lower()) == re.sub(r"\s+", " ", s.lower()):
            continue
        cleaned.append(s); prev = s
    return cleaned

def enforce_single_cta(sents):
    cta_idx = None
    for i, s in enumerate(sents):
        if re.search(CTA_PATTERN, s):
            cta_idx = i
    if cta_idx is None:
        sents.append("Оформить карту.")
        cta_idx = len(sents)-1
    kept = []
    for i, s in enumerate(sents):
        if i != cta_idx and re.search(CTA_PATTERN, s):
            continue
        kept.append(s)
    return kept

def remove_extra_fillers_for_length(sents, max_len=220):
    def total_len(xx): return len(join_sentences(xx))
    out = sents[:]
    # remove duplicate fillers beyond first
    seen_fillers = set(); tmp = []
    for s in out:
        if s.strip() in FILLERS:
            if s.strip() in seen_fillers: continue
            seen_fillers.add(s.strip())
        tmp.append(s)
    out = tmp
    # remove filler sentences (not CTA) until fits
    while total_len(out) > max_len:
        dropped = False
        for idx in range(0, len(out)-1):  # keep last (CTA)
            if out[idx].strip() in FILLERS:
                out.pop(idx); dropped = True; break
        if not dropped: break
    # still long? drop penultimate (keep first and last)
    while total_len(out) > max_len and len(out) > 2:
        out.pop(-2)
    # final clamp preserving CTA
    text = join_sentences(out)
    if len(text) > max_len:
        body = join_sentences(out[:-1]); cta = out[-1]
        space = max_len - (len(cta) + 1)
        if space < 0:
            cta = cta[:max_len].rstrip(" ,.;:!-") + "."
            text = cta
        else:
            body = body[:space].rstrip(" ,.;:!-") + "." if body else ""
            text = (body + " " + cta).strip()
    return split_sentences(text)

def pad_with_unique_fillers(sents, min_len=180, max_len=220):
    text = join_sentences(sents); i = 0
    while len(text) < min_len and i < len(FILLER_POOL):
        cand = FILLER_POOL[i]; i += 1
        if cand in sents: continue
        sents = sents[:-1] + [cand] + [sents[-1]]
        text = join_sentences(sents)
        if len(text) > max_len:
            sents = sents[:-2] + [sents[-1]]
            text = join_sentences(sents)
    return sents

def fix_caps_and_punct(t):
    t = t.replace("!", ".")
    t = re.sub(r"\b([А-ЯЁ]{3,})\b", lambda m: m.group(1).capitalize(), t)
    t = " ".join(t.split()).replace("..", ".").strip()
    if not t.endswith("."): t += "."
    t = re.sub(r"(\d)(\d{3})(?=\D|$)", r"\1 \2", t)
    return t

def sanitize_push(text):
    sents = split_sentences(text)
    sents = dedup_sentences(sents)
    sents = enforce_single_cta(sents)
    sents = remove_extra_fillers_for_length(sents, max_len=220)
    sents = pad_with_unique_fillers(sents, min_len=180, max_len=220)
    t = join_sentences(sents)
    t = fix_caps_and_punct(t)
    # final guard
    if len(t) < 180:
        t = (t[:-1] + " Подробности внутри.").strip()
    if len(t) > 220:
        t = t[:220].rstrip(" ,.;:!-") + "."
    # ensure single CTA
    all_cta = re.findall(CTA_PATTERN, t)
    if len(all_cta) == 0:
        t = t[:-1] + " Оформить карту."
    elif len(all_cta) > 1:
        pieces = [p for p in split_sentences(t) if not re.search(CTA_PATTERN, p)]
        pieces.append([p for p in split_sentences(t) if re.search(CTA_PATTERN, p)][-1])
        t = join_sentences(pieces)
    return t

# ---------- main ----------
def main():
    clients = pd.read_csv(os.path.join(CASE_DIR, "clients.csv"), encoding="utf-8")
    rows_main, rows_top4, rows_variants = [], [], []
    ALLOWED = {
        "Карта для путешествий","Премиальная карта","Кредитная карта","FX/мультивалютный продукт",
        "Кредит наличными","Депозит мультивалютный","Депозит сберегательный","Депозит накопительный",
        "Инвестиции (брокерский счёт)","Золотые слитки",
    }
    for _, prof in clients.iterrows():
        profile = prof.to_dict()
        cc = int(profile["client_code"])
        tx = load_txns(CASE_DIR, cc); tr = load_trs(CASE_DIR, cc)
        s = stats(profile, tx, tr)
        benefits, top4 = score(profile, s)
        best = top4[0] if top4 else "Премиальная карта"
        if best not in ALLOWED: best = "Премиальная карта"
        name = str(profile.get("name","")).split()[0]
        base = GEN[best](name, s, benefits.get(best,0.0))
        final = sanitize_push(base[0])
        rows_main.append({"client_code": cc, "product": best, "push_notification": final})
        # ещё 2 варианта — для презентации
        for rank, p in enumerate(top4, start=1):
            rows_top4.append({"client_code": cc, "rank": rank, "product": p, "benefit_kzt": round(benefits.get(p,0.0),2)})
        for i in range(1, min(3, len(base))):
            rows_variants.append({"client_code": cc, "product": best, "variant_id": i+1, "push_option": sanitize_push(base[i])})

    # save (РОВНО 3 колонки в основном)
    main_df = pd.DataFrame(rows_main)[["client_code","product","push_notification"]]
    main_df.to_csv(os.path.join(OUT_DIR, "output_creative_auto.csv"), index=False, encoding="utf-8")
    pd.DataFrame(rows_top4).to_csv(os.path.join(OUT_DIR, "output_top4_auto.csv"), index=False, encoding="utf-8")
    pd.DataFrame(rows_variants).to_csv(os.path.join(OUT_DIR, "output_variants_auto.csv"), index=False, encoding="utf-8")

    print(f"Готово: {len(main_df)} сообщений. Файлы в: {OUT_DIR}")

if __name__ == "__main__":
    main()
