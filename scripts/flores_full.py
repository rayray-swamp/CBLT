"""FLORES+ 8言語: per-char ダンプ + monotonic θグリッド (Chat確定仕様)。
ダンプ: flores_dumps/flores_dump_<lang>.csv (sent_id,文字,バイト(hex),エントロピー(bits),H,備考, BOS/EOS込, float32)
グリッド: flores_dumps/flores_monotonic_grid.csv
  列: lang,script,bytes_per_char,gold_type,method,theta_bits,theta_nats,bpp,F1,precision,recall,morph_rate,n_patches
仕様: patch start = ΔH(c)=H(c)-H(c-1) > θ_bits, 各文 c=0 強制。H=√ℓ集約(規約B,float32,bits)。
  bpp=総バイト/n_patches(c=0含む)。F1/P/R=patch start文字 vs gold境界文字, 文字単位, c=0除外。
  morph_rate=patch表層がgold1トークンと完全一致するpatchの割合(空白言語は先頭スペース1個除いて照合)。"""
import math, json, csv, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
FP = "/gs/bs/tga-RLA/yoshida/blt_data/floresplus"
THETAS = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
# lang -> (script, gold_type)
LANGMETA = {"en": ("Latn", "ws"), "ru": ("Cyrl", "ws"), "ar": ("Arab", "ws"), "ko": ("Hang", "ws"),
            "hi": ("Deva", "ws"), "ja": ("Jpan", "mecab"), "zh": ("Hans", "jieba"), "th": ("Thai", "pythainlp")}

def gold_spans(text, gtype, tagger=None):
    """gold トークンの (start,end) char-span list。"""
    spans = []
    if gtype == "ws":
        start = None
        for i, c in enumerate(text):
            if c.isspace():
                if start is not None: spans.append((start, i)); start = None
            elif start is None: start = i
        if start is not None: spans.append((start, len(text)))
    elif gtype == "mecab":
        base = 0
        for line in text.split("\n"):
            pos = 0
            for w in tagger(line):
                spans.append((base + pos, base + pos + len(w.surface))); pos += len(w.surface)
            base += len(line) + 1
    elif gtype == "jieba":
        import jieba
        pos = 0
        for w in jieba.cut(text, cut_all=False):
            spans.append((pos, pos + len(w))); pos += len(w)
    elif gtype == "pythainlp":
        from pythainlp import word_tokenize
        pos = 0
        for w in word_tokenize(text, keep_whitespace=True):
            spans.append((pos, pos + len(w))); pos += len(w)
    return spans

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    import fugashi; tagger = fugashi.Tagger()
    ln = math.log(2); boff = 1
    grid_rows = []
    for lang, (script, gtype) in LANGMETA.items():
        sents = [json.loads(l) for l in open(f"{FP}/floresplus_{lang}_devtest.jsonl")]
        dump = open(f"{out_dir}/flores_dump_{lang}.csv", "w", newline="", encoding="utf-8-sig")
        dw = csv.writer(dump); dw.writerow(["sent_id", "文字", "バイト(hex)", "エントロピー(bits)", "H(=sum/√ℓ)", "備考"])
        # θ別アキュムレータ
        acc = {th: dict(npatch=0, nbytes=0, tp=0, npred=0, ngold=0, morph=0) for th in THETAS}
        tot_bytes_per_char = 0; tot_chars = 0
        for d in sents:
            text = d["text"]
            if len(text) < 2: continue
            tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=device).unsqueeze(0)
            e = calculate_entropies(tokens, model, 1, device)[0].float()[0]   # float32
            raw = text.encode("utf-8"); n = len(raw)
            # char分割 + H + ell
            chars = []; bi = 0
            while bi < n:
                ell = 1
                while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
                chars.append((bi, ell)); bi += ell
            C = len(chars)
            H = []; surps_all = []
            for (k, ell) in chars:
                surps = [float(e[k + i]) / ln for i in range(ell)]
                H.append(sum(surps) / math.sqrt(ell)); surps_all.append(surps)
            tot_bytes_per_char += n; tot_chars += C
            # --- dump 書き込み ---
            dw.writerow([d["id"], "<BOS>", "", f"{float(e[0])/ln:.6f}", "", "BOS出力＝先頭バイトを予測(先頭文字Hに算入)"])
            for ci, (k, ell) in enumerate(chars):
                hexs = " ".join(f"{raw[k+i]:02X}" for i in range(ell))
                ents = " ".join(f"{s:.6f}" for s in surps_all[ci])
                note = "先頭文字: 1番目entは<BOS>(e[0])と同値＝BOS項" if ci == 0 else ""
                dw.writerow([d["id"], raw[k:k+ell].decode("utf-8", "replace"), hexs, ents, f"{H[ci]:.6f}", note])
            if n < e.shape[0]: dw.writerow([d["id"], "<EOS>", "", f"{float(e[n])/ln:.6f}", "", "末バイト→EOS"])
            dw.writerow([])
            # --- gold (char-span) ---
            gsp = gold_spans(text, gtype, tagger)
            gspan_set = set(gsp); gstart = set(s for s, _ in gsp)
            # char開始位置(codepoint index)。chars は byte 区切りだが1文字=1 codepoint前提でcidx=codepoint
            # patch境界判定用に各char の codepoint index = ci (0..C-1)
            # --- θ sweep ---
            for th in THETAS:
                pstarts = [0] + [ci for ci in range(1, C) if (H[ci] - H[ci - 1]) > th]
                pstarts = sorted(set(pstarts))
                # patches = (pstarts[i], pstarts[i+1])  char-span
                ends = pstarts[1:] + [C]
                a = acc[th]
                a["npatch"] += len(pstarts); a["nbytes"] += n
                # bpp は総バイト/総patch（後で割る）
                # F1: pred boundary chars (c=0除く) vs gold boundary chars (c=0除く)
                pred = set(pstarts) - {0}; goldb = gstart - {0}
                a["tp"] += len(pred & goldb); a["npred"] += len(pred); a["ngold"] += len(goldb)
                # morph_rate: patch char-span が gold span と完全一致(空白言語は先頭スペース1個除外)
                for ps, pe in zip(pstarts, ends):
                    cs = ps
                    if gtype == "ws" and cs < C and text[cs].isspace(): cs += 1
                    if (cs, pe) in gspan_set: a["morph"] += 1
        dump.close()
        # grid 行
        bpc = tot_bytes_per_char / max(1, tot_chars)
        for th in THETAS:
            a = acc[th]; npatch = max(1, a["npatch"])
            P = a["tp"] / a["npred"] if a["npred"] else 0.0
            R = a["tp"] / a["ngold"] if a["ngold"] else 0.0
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            grid_rows.append([lang, script, f"{bpc:.3f}", gtype, "monotonic",
                              f"{th:.3f}", f"{th*ln:.4f}", f"{a['nbytes']/npatch:.4f}",
                              f"{F1:.4f}", f"{P:.4f}", f"{R:.4f}", f"{a['morph']/npatch:.4f}", a["npatch"]])
        print(f"[OK] {lang}({script},{gtype}): dump + grid")
    with open(f"{out_dir}/flores_monotonic_grid.csv", "w", newline="", encoding="utf-8-sig") as f:
        gw = csv.writer(f)
        gw.writerow(["lang", "script", "bytes_per_char", "gold_type", "method", "theta_bits", "theta_nats",
                     "bpp", "F1", "precision", "recall", "morph_rate", "n_patches"])
        gw.writerows(grid_rows)
    print(f"[OK] grid → flores_monotonic_grid.csv ({len(grid_rows)}行)")

if __name__ == "__main__":
    app()
