"""
70 held-out 文書の per-char CBLT エントロピー H(c)=(Σeᵢ)/√ℓ を算出し、
- 全文字 CSV（doc, lang, pos, char, H_healthy_bits, H_degen_bits）
- 文書ごと PNG バーチャート（healthy teal vs degenerate red）70枚
を出力。GPU 必須。

出力先: /gs/bs/tga-RLA/yoshida/tmp/perchar70/
"""
import math, json, os
import torch
from PIL import Image, ImageDraw, ImageFont

TOKENIZER = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
OFFSET = 3
OUT = "/gs/bs/tga-RLA/yoshida/tmp/perchar70"
HEALTHY = "/gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated"
DEGEN = "/gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/checkpoints/0000102000/consolidated"
JSONL = "/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl"
SEQ = 8192
LN = math.log(2)


def perchar_Hc(model, calc, text, device):
    """text -> list[(char, ell, H(c) bits)] （CBLT √ℓ）"""
    toks = TOK.encode(text)[:SEQ]
    tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    entropies, _ = calc(tokens, model, 1, device)
    e = entropies[0]  # nats per byte (BOS at index0)
    raw = text.encode("utf-8")
    boff = 1  # BOS
    avail = len(toks) - 1  # entropy 有効バイト数（概算）
    out = []
    bi = 0
    while bi < len(raw):
        ell = 1
        while bi + ell < len(raw) and (raw[bi+ell] & 0xC0) == 0x80:
            ell += 1
        if boff + bi + ell > e.shape[0]:
            break
        seg = e[boff+bi: boff+bi+ell]
        h = seg.sum().item() / math.sqrt(ell) / LN
        out.append((raw[bi:bi+ell].decode("utf-8", "replace"), ell, h))
        bi += ell
    return out


def render(path, idx, lang, chars, hh, dd):
    n = len(chars)
    show_labels = n <= 200
    gw = 11 if show_labels else 4
    L, R, T, B = 50, 20, 40, (60 if show_labels else 24)
    W = min(8000, L + R + n * gw); H = 380
    img = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    def font(s):
        # NotoSansJP（en+ja 両対応）を最優先。無ければ dejavu→default。
        for q in ["/gs/bs/tga-RLA/yoshida/fonts/NotoSansJP.otf",
                  "/gs/bs/tga-RLA/yoshida/fonts/NotoSansJP.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(q):
                try: return ImageFont.truetype(q, s)
                except Exception: pass
        return ImageFont.load_default()
    f10 = font(10); f13 = font(13)
    mh = sum(hh)/len(hh) if hh else 0
    dr.text((L, 8), f"doc#{idx} ({lang}) per-char H(c) bits  teal=healthy red=degenerate  mean_h={mh:.2f}", fill=(30,30,30), font=f13)
    ph = H - T - B; ymax = 7.0
    for k in range(0, 8, 1):
        y = T + ph - int(k/ymax*ph)
        dr.line([(L, y), (W-R, y)], fill=(235,235,233))
        dr.text((L-16, y-6), str(k), fill=(150,150,150), font=f10)
    for i in range(n):
        x = L + i*gw
        bw = max(1, gw/2 - 0.5)
        dr.rectangle([x, T+ph-hh[i]/ymax*ph, x+bw, T+ph], fill=(29,158,117))
        dr.rectangle([x+bw, T+ph-dd[i]/ymax*ph, x+2*bw, T+ph], fill=(226,75,74))
        if show_labels:
            dr.text((x, T+ph+4), chars[i], fill=(60,60,60), font=f10)
    img.save(path)


def main():
    global TOK
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
    from bytelatent.data.patcher import calculate_entropies
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT+"/png", exist_ok=True)
    TOK = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOKENIZER}).build()
    docs = [json.loads(l) for l in open(JSONL)]
    dev = "cuda"

    print("loading healthy..."); mh_model,_ = load_entropy_model(HEALTHY, f"{HEALTHY}/consolidated.pth", device=dev); mh_model=mh_model.eval()
    H = [perchar_Hc(mh_model, calculate_entropies, d["text"], dev) for d in docs]
    del mh_model; torch.cuda.empty_cache()
    print("loading degenerate..."); dg_model,_ = load_entropy_model(DEGEN, f"{DEGEN}/consolidated.pth", device=dev); dg_model=dg_model.eval()
    D = [perchar_Hc(dg_model, calculate_entropies, d["text"], dev) for d in docs]
    del dg_model

    # CSV
    with open(OUT+"/perchar_entropy_70.csv", "w", encoding="utf-8") as f:
        f.write("doc_idx,lang,pos,char,ell,H_healthy_bits,H_degenerate_bits\n")
        for i, d in enumerate(docs):
            lang = d.get("lang","?")
            hc = H[i]; dc = D[i]
            m = min(len(hc), len(dc))
            for p in range(m):
                ch, ell, hv = hc[p]; _,_,dv = dc[p]
                cc = ch.replace(",", "<comma>").replace("\n", "<nl>").replace('"','')
                f.write(f"{i},{lang},{p},\"{cc}\",{ell},{hv:.4f},{dv:.4f}\n")
    print("CSV done")

    # PNG x70
    for i, d in enumerate(docs):
        lang = d.get("lang","?")
        hc = H[i]; dc = D[i]; m = min(len(hc), len(dc))
        m = min(m, 150)  # 画像は読みやすさ優先で先頭150文字（全文字はCSV）
        chars = [hc[p][0] for p in range(m)]
        hh = [hc[p][2] for p in range(m)]; dd = [dc[p][2] for p in range(m)]
        render(f"{OUT}/png/doc{i:02d}_{lang}.png", i, lang, chars, hh, dd)
    print("PNG x", len(docs), "done")


if __name__ == "__main__":
    main()
