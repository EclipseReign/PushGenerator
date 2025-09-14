import io, os, re, zipfile, tempfile
import pandas as pd
import streamlit as st

CTA_PATTERN = r"\b(Откр(ойте|ыть)|Оформить|Подключить|Узнать|Проверить|Рассчитать|Настроить|Создать|Купить|Посмотреть|Начать|Разместить)\b"
RU_MONTHS = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}
DEFAULT_LIMITS = {"travel_cashback_cap":15000,"premium_cashback_cap":20000,"credit_card_bonus_cap":25000}

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

FILLERS = {
    "Без лишних шагов.", "Детали внутри.", "Пара касаний — готово.", "Всё прозрачно.",
    "Ничего лишнего.", "Комфорт в поездках и каждый день.", "Курс — под ваш контроль.",
    "Без спешки и случайных курсов.", "Полезно в повседневных расходах.",
    "Сделаете в приложении за минуту.", "Подробности внутри.", "Это быстро и просто.",
    "Всё нужное — в приложении.",
}
FILLER_POOL = ["Подробности внутри.", "Это быстро и просто.", "Всё нужное — в приложении.", "Ничего лишнего."]

ALLOWED_PRODUCTS = {
    "Карта для путешествий","Премиальная карта","Кредитная карта","FX/мультивалютный продукт",
    "Кредит наличными","Депозит мультивалютный","Депозит сберегательный","Депозит накопительный",
    "Инвестиции (брокерский счёт)","Золотые слитки",
}

def fmt_kzt(a):
    try:
        if pd.isna(a): return ""
        return f"{a:,.0f}".replace(",", " ") + " ₸"
    except Exception:
        return ""

def cat_to_phrase(cat):
    return CAT_VERBS.get(cat, f"тратите на {str(cat).lower()}") if isinstance(cat, str) else ""

def verbalize_cats(cat_list):
    phrases = [cat_to_phrase(c) for c in cat_list if c]
    phrases = [p for p in phrases if p]
    if not phrases: return ""
    return phrases[0] if len(phrases)==1 else (f"{phrases[0]} и {phrases[1]}" if len(phrases)==2 else f"{phrases[0]}, {phrases[1]} и {phrases[2]}")

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
        month_name = RU_MONTHS.get(int(tx["dt"].dt.month.max() or 0), "")
    inflows = tr.loc[tr["direction"]=="in","amount"].sum() if len(tr) else 0.0
    outflows= tr.loc[tr["direction"]=="out","amount"].sum() if len(tr) else 0.0
    low_bal = float(profile.get("avg_monthly_balance_KZT",0) or 0) < 100_000
    cash_gap = (outflows>inflows*1.3) or (low_bal and outflows>inflows)
    atm_cnt = int((tr["type"]=="atm_withdrawal").sum()) if len(tr) else 0
    free_cash = max(float(profile.get("avg_monthly_balance_KZT",0) or 0)-50_000,0)
    jcr = sum(spend.get(c,0) for c in ["Ювелирные украшения","Косметика и Парфюмерия","Кафе и рестораны"])
    topcats = [c for c,_ in sorted(spend.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    taxi_trips = int((tx["category"]=="Такси").sum()) if len(tx) else 0
    taxi_spend = spend.get("Такси",0)
    return dict(spend=spend, base_spend=base_spend, online_bundle=online_bundle, fx_activity=fx_activity, fx_volume=fx_volume,
                fx_curr=fx_curr, month=month_name, inflows=inflows, outflows=outflows, cash_gap=cash_gap, atm_cnt=atm_cnt,
                free_cash=free_cash, jcr=jcr, topcats=topcats, taxi_trips=taxi_trips, taxi_spend=taxi_spend, vol_low=True)

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
def benefit_fx(volume, has_fx): return 0.007*volume if has_fx else 0.0
def benefit_mdep(avg, fx): return avg*0.02*(3/12) if (avg>0 and fx) else 0.0
def benefit_sdep(avg, vol_low): return avg*0.06*(3/12) if (avg>0 and vol_low) else 0.0
def benefit_adep(free_cash): return free_cash*0.07*(3/12) if free_cash>0 else 0.0
def benefit_inv(free_cash): return min(3000+0.001*free_cash, 7000) if free_cash>0 else 0.0
def benefit_gold(free_cash): return min(0.002*free_cash, 6000) if free_cash>0 else 0.0

def score(profile, s):
    avg = float(profile.get("avg_monthly_balance_KZT",0) or 0)
    b = {
        "Карта для путешествий": benefit_travel(s["spend"]),
        "Премиальная карта": benefit_premium(s["spend"], s["base_spend"], avg, s["atm_cnt"], s["jcr"]),
        "Кредитная карта": benefit_credit(s["spend"], s["online_bundle"]),
        "FX/мультивалютный продукт": benefit_fx(s["fx_volume"], s["fx_activity"]),
        "Кредит наличными": 5000.0 if s["cash_gap"] else 0.0,
        "Депозит мультивалютный": benefit_mdep(avg, s["fx_activity"]),
        "Депозит сберегательный": benefit_sdep(avg, s["vol_low"]),
        "Депозит накопительный": benefit_adep(s["free_cash"]),
        "Инвестиции (брокерский счёт)": benefit_inv(s["free_cash"]),
        "Золотые слитки": benefit_gold(s["free_cash"]),
    }
    if not s["fx_activity"]:
        b["FX/мультивалютный продукт"] = 0.0
        b["Депозит мультивалютный"] *= 0.5
    top4 = [p for p,_ in sorted(b.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    return b, top4

def copy_credit(name, s, benefit):
    cats_text = verbalize_cats(s["topcats"])
    b = f" — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return f"{name}, мы заметили, что вы часто {cats_text}. Кредитная карта добавит бонусы к привычным покупкам и онлайн-сервисам{b}. {CTA['Кредитная карта'][0]}."
def copy_travel(name, s, benefit):
    taxi = s["taxi_trips"]; ts = fmt_kzt(s["taxi_spend"]); m = s["month"]
    b = f" — сэкономили бы ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return f"{name}, мы заметили поездки в {m}: такси {taxi} раз на {ts}. Тревел-карта вернёт часть расходов{b}. {CTA['Карта для путешествий'][0]}."
def copy_premium(name, s, benefit):
    b = f" — до {fmt_kzt(benefit)}" if benefit>0 else ""
    return f"{name}, траты в ресторанах и высокий остаток — повод получать больше. Премиальная карта добавит кешбэк и комфорт в поездках{b}. {CTA['Премиальная карта'][0]}."
def copy_fx(name, s, benefit):
    curr = s["fx_curr"]; b = f" — сэкономите на спреде ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return f"{name}, оплачиваете в {curr}? Включите обмен заранее и авто-покупку по целевому курсу{b}. {CTA['FX/мультивалютный продукт'][0]}."
def copy_deposit(name, s, benefit, prod):
    b = f" — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else ""
    return f"{name}, закрепите часть средств на вкладе — копить так проще и спокойнее{b}. {CTA[prod][0]}."
def copy_cash_loan(name, s, benefit): return f"{name}, если намечаются крупные траты, кредит наличными сгладит платежи и добавит гибкости. {CTA['Кредит наличными'][0]}."
def copy_invest(name, s, benefit): return f"{name}, начните аккуратно: счёт открывается за пару касаний, без лишней сложности. {CTA['Инвестиции (брокерский счёт)'][0]}."

def base_text(product, name, s, benefits):
    mapping = {
        "Кредитная карта": copy_credit,
        "Карта для путешествий": copy_travel,
        "Премиальная карта": copy_premium,
        "FX/мультивалютный продукт": copy_fx,
        "Депозит мультивалютный": lambda n,ss,bb: copy_deposit(n,ss,bb,"Депозит мультивалютный"),
        "Депозит сберегательный": lambda n,ss,bb: copy_deposit(n,ss,bb,"Депозит сберегательный"),
        "Депозит накопительный": lambda n,ss,bb: copy_deposit(n,ss,bb,"Депозит накопительный"),
        "Кредит наличными": copy_cash_loan,
        "Инвестиции (брокерский счёт)": copy_invest,
        "Золотые слитки": lambda n,ss,bb: f"{n}, добавьте немного золота для баланса сбережений. {CTA['Золотые слитки'][0]}.",
    }
    return mapping[product](name, s, benefits.get(product,0.0))

def split_sentences(text):
    t = " ".join(str(text).split()).replace("..", ".").strip()
    if not t.endswith("."): t += "."
    parts = [s.strip() for s in re.split(r"\.\s*", t) if s.strip()]
    return [p if p.endswith(".") else p+"." for p in parts]

def join_sentences(sents):
    out = " ".join(s.strip() for s in sents if s and s.strip())
    out = re.sub(r"\s+"," ",out).strip().replace("..",".")
    if not out.endswith("."): out+="."
    return out

def dedup_sentences(sents):
    seen=set(); out=[]
    for s in sents:
        key=re.sub(r"\s+"," ",s.strip().lower())
        if key in seen: continue
        seen.add(key); out.append(s)
    cleaned=[]; prev=None
    for s in out:
        if prev and re.sub(r"\s+"," ",prev.lower())==re.sub(r"\s+"," ",s.lower()): continue
        cleaned.append(s); prev=s
    return cleaned

def enforce_single_cta(sents):
    cta_idx=None
    for i,s in enumerate(sents):
        if re.search(CTA_PATTERN,s): cta_idx=i
    if cta_idx is None:
        sents.append("Оформить карту."); cta_idx=len(sents)-1
    kept=[]
    for i,s in enumerate(sents):
        if i!=cta_idx and re.search(CTA_PATTERN,s): continue
        kept.append(s)
    return kept

def remove_extra_fillers_for_length(sents, max_len=220):
    def total_len(xx): return len(join_sentences(xx))
    out=sents[:]
    seen_fill=set(); tmp=[]
    for s in out:
        if s.strip() in FILLERS:
            if s.strip() in seen_fill: continue
            seen_fill.add(s.strip())
        tmp.append(s)
    out=tmp
    while total_len(out)>max_len:
        dropped=False
        for idx in range(0,len(out)-1):
            if out[idx].strip() in FILLERS:
                out.pop(idx); dropped=True; break
        if not dropped: break
    while total_len(out)>max_len and len(out)>2: out.pop(-2)
    text=join_sentences(out)
    if len(text)>max_len:
        body=join_sentences(out[:-1]); cta=out[-1]
        space=max_len-(len(cta)+1)
        if space<0: text=cta[:max_len].rstrip(" ,.;:!-")+"."
        else:
            body=body[:space].rstrip(" ,.;:!-")+"." if body else ""
            text=(body+" "+cta).strip()
    return split_sentences(text)

def pad_with_unique_fillers(sents, min_len=180, max_len=220):
    text=join_sentences(sents); i=0
    while len(text)<min_len and i<len(FILLER_POOL):
        cand=FILLER_POOL[i]; i+=1
        if cand in sents: continue
        sents=sents[:-1]+[cand]+[sents[-1]]
        text=join_sentences(sents)
        if len(text)>max_len:
            sents=sents[:-2]+[sents[-1]]
            text=join_sentences(sents)
    return sents

def fix_caps_and_punct(t):
    t=t.replace("!",".")
    t=re.sub(r"\b([А-ЯЁ]{3,})\b",lambda m:m.group(1).capitalize(),t)
    t=" ".join(t.split()).replace("..",".").strip()
    if not t.endswith("."): t+="."
    t=re.sub(r"(\d)(\d{3})(?=\D|$)", r"\1 \2", t)
    return t

def ensure_value(product, s, benefit, text):
    has_value = any(k in text.lower() for k in ["кешбэк","выгода","сэконом", "без комиссий","до 10%","до 4%","ставк","процент","бесплатные снятия"])
    if has_value:
        return text
    value_map = {
        "Кредитная карта": f"До 10% в любимых категориях и на онлайн-сервисы — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else "До 10% в любимых категориях и на онлайн-сервисы",
        "Карта для путешествий": f"Кешбэк на билеты, отели и такси — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else "Кешбэк на билеты, отели и такси",
        "Премиальная карта": f"Повышенный кешбэк и бесплатные снятия — до {fmt_kzt(benefit)}" if benefit>0 else "Повышенный кешбэк и бесплатные снятия",
        "FX/мультивалютный продукт": f"Автопокупка по целевому курсу, экономия на спреде ≈{fmt_kzt(benefit)}" if benefit>0 else "Автопокупка по целевому курсу",
        "Депозит мультивалютный": f"Процент на остаток в валюте — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else "Процент на остаток в валюте",
        "Депозит сберегательный": f"Ставка до 6% годовых — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else "Ставка до 6% годовых",
        "Депозит накопительный": f"Копите автоматически — выгода ≈{fmt_kzt(benefit)}" if benefit>0 else "Копите автоматически",
        "Кредит наличными": "Гибкий график платежей и прозрачные условия",
        "Инвестиции (брокерский счёт)": "Первые шаги без лишних комиссий",
        "Золотые слитки": "Защитный актив для сохранения стоимости",
    }
    ins = value_map.get(product, "")
    sents = split_sentences(text)
    if len(sents) >= 1:
        sents = sents[:-1] + [ins if ins.endswith(".") else ins + "."] + [sents[-1]]
    return join_sentences(sents)

def sanitize_push_with_value(product, s, benefit, text):
    sents = split_sentences(text)
    sents = dedup_sentences(sents)
    sents = enforce_single_cta(sents)
    sents = remove_extra_fillers_for_length(sents, 220)
    sents = pad_with_unique_fillers(sents, 180, 220)
    t = fix_caps_and_punct(join_sentences(sents))
    t = ensure_value(product, s, benefit, t)
    if len(t) < 180:
        t = (t[:-1] + " Подробности внутри.").strip()
    if len(t) > 220:
        t = t[:220].rstrip(" ,.;:!-") + "."
    all_cta = re.findall(CTA_PATTERN, t)
    if len(all_cta) == 0:
        t = t[:-1] + " Оформить карту."
    elif len(all_cta) > 1:
        body=[p for p in split_sentences(t) if not re.search(CTA_PATTERN,p)]
        cta=[p for p in split_sentences(t) if re.search(CTA_PATTERN,p)][-1]
        t = join_sentences(body+[cta])
    return t

def find_case_dir(root_dir):
    candidates = []
    for r, d, files in os.walk(root_dir):
        if "clients.csv" in files:
            tx = sum(1 for f in files if re.match(r"client_\d+_transactions_3m\.csv", f, re.I))
            tr = sum(1 for f in files if re.match(r"client_\d+_transfers_3m\.csv", f, re.I))
            score = tx + tr
            candidates.append((score, r))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]

def generate(case_dir):
    errors, warnings = [], []
    clients_path = os.path.join(case_dir, "clients.csv")
    if not os.path.exists(clients_path):
        errors.append("Не найден файл clients.csv.")
        return None, None, None, errors, warnings
    try:
        clients = pd.read_csv(clients_path, encoding="utf-8")
    except Exception as e:
        errors.append(f"clients.csv не читается: {e}")
        return None, None, None, errors, warnings
    if "client_code" not in clients.columns:
        errors.append("В clients.csv отсутствует колонка client_code.")
        return None, None, None, errors, warnings
    if clients.empty:
        warnings.append("clients.csv пустой — результат будет пустым.")

    missing_tx, missing_tr = [], []
    for _, prof in clients.iterrows():
        cc = int(prof["client_code"])
        if not os.path.exists(os.path.join(case_dir, f"client_{cc}_transactions_3m.csv")):
            missing_tx.append(cc)
        if not os.path.exists(os.path.join(case_dir, f"client_{cc}_transfers_3m.csv")):
            missing_tr.append(cc)
    if missing_tx:
        warnings.append(f"Нет transactions-файлов для клиентов: {missing_tx[:20]}{' …' if len(missing_tx)>20 else ''}")
    if missing_tr:
        warnings.append(f"Нет transfers-файлов для клиентов: {missing_tr[:20]}{' …' if len(missing_tr)>20 else ''}")

    rows_main, rows_top4, rows_vars = [], [], []
    for _, prof in clients.iterrows():
        profile = prof.to_dict()
        cc = int(profile["client_code"])
        tx = load_txns(case_dir, cc); tr = load_trs(case_dir, cc)
        s = stats(profile, tx, tr)
        benefits, top4 = score(profile, s)
        best = top4[0] if top4 else "Премиальная карта"
        if best not in ALLOWED_PRODUCTS: best = "Премиальная карта"
        name = str(profile.get("name","")).split()[0]
        base = base_text(best, name, s, benefits)
        final = sanitize_push_with_value(best, s, benefits.get(best,0.0), base)
        if not (180 <= len(final) <= 220):  # финальный коридор
            final = sanitize_push_with_value(best, s, benefits.get(best,0.0), final)
        rows_main.append({"client_code": cc, "product": best, "push_notification": final})
        for rank, p in enumerate(top4, start=1):
            rows_top4.append({"client_code": cc, "rank": rank, "product": p, "benefit_kzt": round(float(benefits.get(p,0.0)),2)})

        alt_cta = {"Кредитная карта":"Проверить лимит","Карта для путешествий":"Открыть карту"}.get(best)
        if alt_cta:
            alt = final.replace("Оформить карту.", alt_cta + ".")
            alt = sanitize_push_with_value(best, s, benefits.get(best,0.0), alt)
            rows_vars.append({"client_code": cc, "product": best, "variant_id": 2, "push_option": alt})

    df_main = pd.DataFrame(rows_main)[["client_code","product","push_notification"]]
    df_top4 = pd.DataFrame(rows_top4)
    df_vars = pd.DataFrame(rows_vars) if rows_vars else pd.DataFrame(columns=["client_code","product","variant_id","push_option"])
    return df_main, df_top4, df_vars, errors, warnings

st.set_page_config(page_title="Push Generator — EclipseReign", layout="centered")

st.markdown("""
<style>
.credit-card {
  padding:14px 16px;border-radius:16px;
  background: linear-gradient(135deg,#111827 0%,#1f2937 60%,#0ea5e9 120%);
  color:#f8fafc; box-shadow:0 12px 28px rgba(0,0,0,.35);
  border:1px solid rgba(255,255,255,.08);
}
.credit-title{font-weight:800;font-size:18px;letter-spacing:.3px;margin:0 0 6px 0;}
.credit-badge{display:inline-block;margin-bottom:10px;padding:2px 10px;border-radius:999px;background:rgba(255,255,255,.12);font-size:12px}
.credit-item{margin:6px 0 0 0;font-size:14px}
.credit-role{opacity:.85}
.credit-name{font-weight:700}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
<div class="credit-card">
  <div class="credit-title">EclipseReign</div>
  <div class="credit-badge">Hackathon Team</div>
  <div class="credit-item"><span class="credit-role">Full-Stack Developer</span> — <span class="credit-name">Revenko Denis</span></div>
  <div class="credit-item"><span class="credit-role">Data Analyst</span> — <span class="credit-name">Aiganym Tulebayeva</span></div>
  <div class="credit-item"><span class="credit-role">Designers</span> — <span class="credit-name">Alibek Zharmuhambetuly</span>, <span class="credit-name">Zhibek Mussakulova</span></div>
</div>
        """, unsafe_allow_html=True
    )

st.title("Push Generator")
st.caption("Загрузите датасет (ZIP с clients.csv внутри или набор CSV) → нажмите «Сгенерировать пуши» → скачайте результат.")

mode = st.radio("Формат загрузки", ["ZIP архив", "CSV-файлы"], horizontal=True)

zip_file = None
csv_files = []
if mode == "ZIP архив":
    zip_file = st.file_uploader("Загрузите ZIP", type=["zip"])
else:
    st.markdown("Минимум нужен **clients.csv**. Дополнительно: `client_{N}_transactions_3m.csv`, `client_{N}_transfers_3m.csv`.")
    csv_files = st.file_uploader("Загрузите CSV (можно несколько)", type=["csv"], accept_multiple_files=True)

run = st.button("Сгенерировать пуши", type="primary", disabled=(mode=="ZIP архив" and zip_file is None) or (mode!="ZIP архив" and not csv_files))

if run:
    errors, warnings = [], []
    with st.spinner("Обработка..."):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                case_dir = None
                if zip_file is not None:
                    zpath = os.path.join(tmp, "upload.zip")
                    with open(zpath, "wb") as f: f.write(zip_file.getbuffer())
                    with zipfile.ZipFile(zpath, "r") as zf: zf.extractall(tmp)
                    case_dir = find_case_dir(tmp)
                    if not case_dir:
                        errors.append("В ZIP не найден ни один каталог с clients.csv.")
                else:
                    case_dir = os.path.join(tmp, "dataset")
                    os.makedirs(case_dir, exist_ok=True)
                    for uf in csv_files:
                        if not uf.name.lower().endswith(".csv"):
                            warnings.append(f"Пропущен не-CSV файл: {uf.name}")
                            continue
                        with open(os.path.join(case_dir, os.path.basename(uf.name)), "wb") as f:
                            f.write(uf.getbuffer())

                if not errors and case_dir:
                    df_main, df_top4, df_vars, errs, warns = generate(case_dir)
                    errors.extend(errs); warnings.extend(warns)
        except zipfile.BadZipFile:
            errors.append("Файл не является корректным ZIP.")
        except Exception as e:
            errors.append(f"Непредвиденная ошибка: {e}")

    if warnings:
        for w in warnings: st.warning(w)
    if errors:
        for e in errors: st.error(e)
    else:
        st.success("Готово!")

        st.subheader("Итог (на сдачу)")
        st.caption("Ровно 3 колонки: client_code, product, push_notification")
        st.dataframe(df_main.head(20), use_container_width=True, hide_index=True)

        st.subheader("Top-4 на клиента (объяснимость)")
        st.dataframe(df_top4.head(20), use_container_width=True, hide_index=True)

        if len(df_vars):
            st.subheader("Варианты (для презентации)")
            st.dataframe(df_vars.head(20), use_container_width=True, hide_index=True)

        #def qa_check(text):
        #    issues=[]
        #    if not (180 <= len(text) <= 220): issues.append("length")
        #    if len(re.findall(CTA_PATTERN, text)) != 1: issues.append("cta_count")
        #    if "!" in text: issues.append("exclamations")
        #    if re.search(r"\b[А-ЯЁ]{3,}\b", text): issues.append("all_caps")
        #    if "₸" in text and not re.search(r"\d{1,3}( \d{3})* ₸", text): issues.append("currency")
        #    if not any(k in text.lower() for k in ["кешбэк","выгода","сэконом","без комиссий","до 10%","до 4%","ставк","процент","бесплатные снятия"]):
        #        issues.append("no_value")
        #    return issues or ["OK"]
        #st.subheader("QA по ТЗ")
        #sample = df_main["push_notification"].tolist()
        #if sample:
        #    lens = [len(t) for t in sample]
        #    st.write(f"Проверено сообщений: {len(sample)}. Длина (мин/ср/макс): **{min(lens)} / {int(sum(lens)/len(lens))} / {max(lens)}**")
        #    bad = [(i, qa_check(t)) for i,t in enumerate(sample) if qa_check(t)!=["OK"]]
        #    if bad:
        #        st.warning(f"Найдено несоответствий: {len(bad)} (первые 10 ниже).")
        #        st.json(bad[:10])
        #    else:
        #        st.success("Все сообщения укладываются в ТЗ и содержат явную ценность.")
        #def dl(df, name):
        #    buf = io.StringIO(); df.to_csv(buf, index=False, encoding="utf-8")
        #    st.download_button(f"Download {name}", buf.getvalue().encode("utf-8"), file_name=name, mime="text/csv")
        #dl(df_main, "output_creative_auto.csv")
        #dl(df_top4, "output_top4_auto.csv")
        #if len(df_vars): dl(df_vars, "output_variants_auto.csv")