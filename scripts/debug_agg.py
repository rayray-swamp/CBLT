"""「医」を含む文で aggregate_char_entropy のループを手動再現し、char境界・i/j/ell・out配置・e_sum を
実 aggregate の out と突き合わせる。e と tokens の長さ整合も確認。"""
import math, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy, OFFSET

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
TEXT = "万一大量に飲み込んだ時は、水を飲ませる等の処置を行い、医師に相談してください。"

@app.command()
def main(ckpt_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2)
    toks = tok.encode(TEXT)
    tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    ent = calculate_entropies(tokens, model, 1, device)[0]
    out = aggregate_char_entropy(ent, tokens, "sqrt")[0]
    e = ent[0]
    print(f"len(toks)={len(toks)}  ent.shape={tuple(ent.shape)}  e.shape={tuple(e.shape)}  out.shape={tuple(out.shape)}")
    rb = (tokens[0] - OFFSET)
    is_lead = ((rb & 0xC0) != 0x80)
    # 手動でaggregateループ再現（医=token82付近）し、out配置を確認
    print("\n=== token 79..86 周辺 (医=byte81=token82) ===")
    for t in range(79, 87):
        b = int(rb[t]); bs = f"{b:02X}" if b >= 0 else f"SP({b})"
        o = float(out[t]); os = f"{o/ln:.3f}" if math.isfinite(o) else "-inf"
        print(f"  tok{t}: raw={bs} lead={bool(is_lead[t])} e={float(e[t])/ln:.3f} out={os}")
    # 医(byte81)について: aggregate式 i=82 → out[81]=entropies[81:84]/√3
    print("\n=== 医(byte81)の検算 ===")
    print(f"  e[81:84] (bits) = {[round(float(e[t])/ln,3) for t in range(81,84)]}  sum/√3 = {float(e[81:84].sum())/math.sqrt(3)/ln:.4f}")
    print(f"  e[80:83] (bits) = {[round(float(e[t])/ln,3) for t in range(80,83)]}  sum/√3 = {float(e[80:83].sum())/math.sqrt(3)/ln:.4f}")
    print(f"  e[82:85] (bits) = {[round(float(e[t])/ln,3) for t in range(82,85)]}  sum/√3 = {float(e[82:85].sum())/math.sqrt(3)/ln:.4f}")
    print(f"  実 out[81] = {float(out[81])/ln:.4f}  out[80]={float(out[80])/ln if math.isfinite(float(out[80])) else float('nan'):.4f}  out[82]={float(out[82])/ln if math.isfinite(float(out[82])) else float('nan'):.4f}")
    # 全char走査して out[k] と e[k:k+ell]/√ell が食い違うcharを列挙
    print("\n=== 食い違うchar一覧 (|Δ|>0.001) ===")
    raw = TEXT.encode("utf-8"); n = len(raw); bi = 0; boff = 1
    while bi < n:
        ell = 1
        while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
        He = float(out[bi]) / ln
        Hi = float(e[bi:bi + ell].sum()) / math.sqrt(ell) / ln
        if abs(He - Hi) > 0.001:
            ch = raw[bi:bi + ell].decode("utf-8", "replace")
            print(f"  byte{bi} '{ch}' ell={ell}: out[{bi}]={He:.3f} vs e[{bi}:{bi+ell}]/√{ell}={Hi:.3f}  Δ={He-Hi:+.3f}")
        bi += ell

if __name__ == "__main__":
    app()
