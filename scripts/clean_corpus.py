"""largecorpus データクリーニング（Chat確定仕様, 順序 ①→③→②→④）。
① valid UTF-8: lone-surrogate除去 + U+FFFD率>0.5%除去（率ベース）
③ 短(<50字)/boilerplate(非letter率>70%)/lang-ID(script-based, ja/zh はHan優勢でどちらも整合)
② exact dedup: 正規化(NFC+空白圧縮)は★ハッシュキー計算のみ。残すtextは生原文
④ FLORES+汚染: 両側正規化(NFC+lowercase+空白圧縮)→文単位一致(Aho-Corasick・full sentence)で除去
ガードレール: 言語別 before/after・各フィルタ除去数・FLORES+汚染率・最長一致長分布 をログ。
出力は生原文を保持した jsonl。"""
import json, hashlib, unicodedata, re, time, typer
from collections import defaultdict, Counter
import ahocorasick

app = typer.Typer()
CORPUS = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus/corpus/corpus.chunk.00.jsonl"
FP = "/gs/bs/tga-RLA/yoshida/blt_data/floresplus"
OUT = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus_clean/corpus/corpus.chunk.00.jsonl"
LANGS = ["en", "ru", "ja", "ar", "zh", "ko", "hi", "th"]
WS = re.compile(r"\s+")

def char_script(o):
    if 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xC0 <= o <= 0x24F or 0x1E00 <= o <= 0x1EFF: return "LATIN"
    if 0x400 <= o <= 0x4FF: return "CYRILLIC"
    if 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFDFF: return "ARABIC"
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF: return "HANGUL"
    if 0x900 <= o <= 0x97F: return "DEVANAGARI"
    if 0xE00 <= o <= 0xE7F: return "THAI"
    if 0x3040 <= o <= 0x309F: return "HIRAGANA"
    if 0x30A0 <= o <= 0x30FF: return "KATAKANA"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF: return "CJK"
    return None

EXP = {"en": "LATIN", "ru": "CYRILLIC", "ar": "ARABIC", "ko": "HANGUL", "hi": "DEVANAGARI", "th": "THAI"}

def nonletter_ratio(text):
    s = text[:2000]
    if not s: return 1.0
    letters = sum(1 for c in s if char_script(ord(c)) is not None)
    return 1.0 - letters / len(s)

LANGID_MIN = 0.10  # 期待script率がこれ未満なら誤ラベルとみなし除去（安全側に低め）

def lang_ok(text, lang):
    # ★最頻でなく「期待scriptの合算比率」で判定。日本語は CJK/Hiragana/Katakana に分散するため
    #   最頻だと Latin が勝ち正規docを誤除去する。期待率<10%(=ほぼ無し)のみ誤ラベルとして除去。
    cnt = Counter(); total = 0
    for c in text[:2000]:
        s = char_script(ord(c))
        if s: cnt[s] += 1; total += 1
    if total == 0: return False  # letter が皆無
    if lang == "ja": exp = ("HIRAGANA", "KATAKANA", "CJK")  # ja/zh は Han 共有(Chat指定)
    elif lang == "zh": exp = ("CJK",)
    elif lang in EXP: exp = (EXP[lang],)
    else: return True  # None/未知lang は保持
    return sum(cnt[s] for s in exp) / total >= LANGID_MIN

@app.command()
def main(limit: int = typer.Option(0, help="0=全件")):
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # FLORES+ automaton (正規化済 full sentence)
    A = ahocorasick.Automaton()
    flores_total = 0; flores_by_lang = Counter()
    for lg in LANGS:
        for line in open(f"{FP}/floresplus_{lg}_devtest.jsonl"):
            s = json.loads(line)["text"]
            mk = WS.sub(" ", unicodedata.normalize("NFC", s).lower()).strip()
            if len(mk) >= 20:  # 極短FLORES文は誤検出源なので除外
                A.add_word(mk, (lg, mk)); flores_total += 1; flores_by_lang[lg] += 1
    A.make_automaton()
    print(f"FLORES+ patterns: {flores_total} ({dict(flores_by_lang)})")

    before = Counter(); after = Counter()
    removed = {f: Counter() for f in ["utf8", "short", "boilerplate", "langid", "dedup", "flores"]}
    seen = set(); flores_hit = set(); match_lens = Counter()
    t0 = time.time(); n = 0
    out = open(OUT, "w", encoding="utf-8")
    for line in open(CORPUS):
        n += 1
        if limit and n > limit: break
        try: d = json.loads(line)
        except Exception: continue
        lang = d.get("lang") or "none"; text = d.get("text", "")
        before[lang] += 1
        # ① UTF-8
        try: text.encode("utf-8")
        except UnicodeEncodeError: removed["utf8"][lang] += 1; continue
        if len(text) and text.count("�") / len(text) > 0.005: removed["utf8"][lang] += 1; continue
        # ③ 短
        if len(text) < 50: removed["short"][lang] += 1; continue
        # ③ boilerplate
        if nonletter_ratio(text) > 0.70: removed["boilerplate"][lang] += 1; continue
        # ③ lang-ID
        if not lang_ok(text, lang): removed["langid"][lang] += 1; continue
        # ② dedup（正規化はキー計算のみ・残すのは生text）
        nfc = unicodedata.normalize("NFC", text)
        hk = hashlib.sha1(WS.sub(" ", nfc).strip().encode("utf-8")).digest()
        if hk in seen: removed["dedup"][lang] += 1; continue
        seen.add(hk)
        # ④ FLORES+ 汚染（文単位一致）
        mk = WS.sub(" ", nfc.lower()).strip()
        ms = list(A.iter(mk))
        if ms:
            removed["flores"][lang] += 1
            for _, (flg, fs) in ms: flores_hit.add(fs); match_lens[len(fs)] += 1
            continue
        # survive: 生原文を保持
        out.write(json.dumps({"text": text, "lang": d.get("lang"), "source": d.get("source")}, ensure_ascii=False) + "\n")
        after[lang] += 1
        if n % 1_000_000 == 0: print(f"  {n:,} docs ({time.time()-t0:.0f}s)")
    out.close()

    # ===== ログ =====
    print(f"\n=== クリーニング完了: {n:,} docs 処理 ({time.time()-t0:.0f}s) ===")
    print(f"{'lang':5} {'before':>9} {'after':>9} | utf8 short boiler langid dedup flores")
    for lg in LANGS + ["none"]:
        if before[lg] == 0: continue
        r = [removed[f][lg] for f in ["utf8", "short", "boilerplate", "langid", "dedup", "flores"]]
        print(f"{lg:5} {before[lg]:>9,} {after[lg]:>9,} | {r[0]:,} {r[1]:,} {r[2]:,} {r[3]:,} {r[4]:,} {r[5]:,}")
    print(f"{'計':5} {sum(before.values()):>9,} {sum(after.values()):>9,} | "
          + " ".join(f"{sum(removed[f].values()):,}" for f in ["utf8", "short", "boilerplate", "langid", "dedup", "flores"]))
    print(f"\nFLORES+ 汚染: 除去doc {sum(removed['flores'].values()):,} / corpusでヒットしたFLORES+文 {len(flores_hit)}/{flores_total} "
          f"(contamination rate {len(flores_hit)/max(1,flores_total)*100:.2f}%)")
    print(f"最長一致長分布(文字数): " + " ".join(f"{k}:{v}" for k, v in sorted(match_lens.items())[:20]))
    print(f"出力: {OUT}")

if __name__ == "__main__":
    app()
