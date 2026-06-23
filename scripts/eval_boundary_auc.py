"""
構造eval: 境界整合 AUC（閾値不要・モデル間/段間比較用）

per-char CBLT √ℓ エントロピー H(c) を「境界らしさスコア」とし、gold 境界との AUC を測る。
= 実 patching 信号そのものが言語境界をどれだけ捉えているかを定量化。

gold 境界（char が token 先頭か）:
  ja: MeCab(fugashi) 形態素境界（主役）
  en/ru: 空白区切り（サニティ、高AUC期待）
  zh: jieba（無空白CJK補助）
スコア: H(c) = (Σ e_i)/√ℓ bits（実 calculate_entropies 経路、perchar と同一）
規約: entropy は「次バイト予測」。char c の予測しやすさ = H(c)。境界(=新語頭)char は予測困難 → H(c)高 を期待。
warm-up: 先頭 WARMUP 文字を除外（初期位置のエントロピー乱れ排除）。

GPU 必須。held-out テキスト（全段同一）で実行。
使い方: python scripts/eval_boundary_auc.py --ckpt-dir <consolidated> --jsonl heldout_v2.jsonl --label stepXXXX
"""
import math, json
import typer
import torch
import numpy as np

app = typer.Typer()
TOKENIZER = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
WARMUP_CHARS = 160   # ~ sliding_window 512B / ja3B ≈ 170; en は文字=バイトなので512だが160で十分手前を除外


def gold_boundaries(text, lang, _ja=[None], _zh=[None]):
    """char index の集合（token 先頭 = 境界）を返す"""
    bset=set(); pos=0
    if lang=="ja":
        if _ja[0] is None:
            import fugashi; _ja[0]=fugashi.Tagger()
        for w in _ja[0](text):
            bset.add(pos); pos+=len(w.surface)
    elif lang=="zh":
        if _zh[0] is None:
            import jieba; _zh[0]=jieba
        for w in _zh[0].cut(text):
            bset.add(pos); pos+=len(w)
    else:  # en/ru 等：空白区切り。空白の次の非空白が語頭
        i=0; n=len(text); newword=True
        while i<n:
            if text[i].isspace():
                newword=True
            else:
                if newword: bset.add(i)
                newword=False
            i+=1
    return bset


def auc(scores, labels):
    """rank-based AUC（Mann-Whitney）。sklearn不要"""
    s=np.asarray(scores,dtype=np.float64); y=np.asarray(labels,dtype=np.int8)
    npos=int(y.sum()); nneg=len(y)-npos
    if npos==0 or nneg==0: return float("nan"), npos, nneg
    order=np.argsort(s,kind="mergesort")
    ranks=np.empty(len(s),dtype=np.float64); ranks[order]=np.arange(1,len(s)+1)
    # tie 平均ランク
    s_sorted=s[order]; i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s_sorted[j+1]==s_sorted[i]: j+=1
        if j>i:
            avg=(ranks[order[i]]+ranks[order[j]])/2
            for k in range(i,j+1): ranks[order[k]]=avg
        i=j+1
    sum_pos=ranks[y==1].sum()
    a=(sum_pos - npos*(npos+1)/2)/(npos*nneg)
    return float(a), npos, nneg


@app.command()
def main(ckpt_dir: str = typer.Option(...), jsonl: str = typer.Option(...),
         label: str = typer.Option("model"), device: str = typer.Option("cuda")):
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
    from bytelatent.data.patcher import calculate_entropies
    model,_=load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model=model.eval()
    tok=TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path":TOKENIZER}).build()
    ln=math.log(2)
    per_lang={}  # lang -> (scores[], labels[])
    for d in (json.loads(l) for l in open(jsonl)):
        lang=d.get("lang","en"); text=d["text"]
        toks=tok.encode(text)[:8192]
        if len(toks)<WARMUP_CHARS+64: continue
        tokens=torch.tensor(toks,dtype=torch.long,device=device).unsqueeze(0)
        ent,_=calculate_entropies(tokens,model,1,device)
        e=ent[0]; raw=text.encode("utf-8"); boff=1
        # per-char H(c)
        chars=[]; hc=[]; bi=0
        while bi<len(raw):
            ell=1
            while bi+ell<len(raw) and (raw[bi+ell]&0xC0)==0x80: ell+=1
            if boff+bi+ell>e.shape[0]: break
            # entropies[i] = H(predict token i+1)。char c のバイト(token boff+bi..)を予測する
            # エントロピーは直前位置の出力 = e[boff+bi-1 .. boff+bi+ell-2]（off-by-one 修正）
            h=e[boff+bi-1:boff+bi+ell-1].sum().item()/math.sqrt(ell)/ln
            chars.append(raw[bi:bi+ell].decode("utf-8","replace")); hc.append(h); bi+=ell
        ctext="".join(chars)
        bset=gold_boundaries(ctext,lang)
        sc=per_lang.setdefault(lang,([],[]))
        for ci in range(len(chars)):
            if ci<WARMUP_CHARS: continue
            sc[0].append(hc[ci]); sc[1].append(1 if ci in bset else 0)

    print(f"\n##### label={label} ckpt={ckpt_dir} #####")
    print("lang,n_chars,boundary_rate,AUC")
    for lang,(scores,labels) in sorted(per_lang.items()):
        a,npos,nneg=auc(scores,labels)
        print(f"{lang},{len(labels)},{npos/max(1,len(labels)):.3f},{a:.4f}")


if __name__=="__main__":
    app()
