"""恒久リグレッションテスト: CBLT集約の lead-1 アライメント（規約B）。
off-by-one（規約A）が二度と静かに戻らないよう、worked example を固定値で検証する。
実関数 aggregate_char_entropy を import。PASS で exit 0。
run: gpu_1（patcher import が xformers を引くため計算ノードで）。"""
import math, torch
from bytelatent.data.patcher import aggregate_char_entropy
from bytelatent.tokenizers.constants import OFFSET

def main():
    text = "I have a りんご!"                  # 13文字 / 19バイト, 「り」=bytes[9:12]
    raw = list(text.encode("utf-8"))
    tokens = torch.tensor([[b + OFFSET for b in raw]])
    ent = torch.tensor([[float(i) for i in range(len(raw))]])  # entropy[i]=i → 足したindexが値に出る
    out = aggregate_char_entropy(ent, tokens, "sqrt")[0]
    r3 = math.sqrt(3)
    inf = float("-inf")

    checks = [
        # 核心: 「り」(lead@9) out[8] は 規約B=Σentropies[8,9,10]=27 → 27/√3。規約A(30/√3)ではない
        ("り out[8] == 規約B 27/√3", abs(float(out[8]) - 27/r3) < 1e-3),
        ("り out[8] != 規約A 30/√3(旧バグ)", abs(float(out[8]) - 30/r3) > 1e-3),
        ("ん out[11] == 規約B 36/√3", abs(float(out[11]) - 36/r3) < 1e-3),
        ("ご out[14] == 規約B 45/√3", abs(float(out[14]) - 45/r3) < 1e-3),
        # ASCII: 'h'(lead@2) out[1]=entropies[1]=1, 'a'(lead@3) out[2]=2
        ("h out[1] == 1.0", abs(float(out[1]) - 1.0) < 1e-3),
        ("a out[2] == 2.0", abs(float(out[2]) - 2.0) < 1e-3),
        # 文字保護: continuation byte(10,11,13,14,16,17)で patch開始不可
        # = それを担う out位置 k(=byte位置-1) {9,10,12,13,15,16} が -inf
        ("文字保護: out[9,10,12,13,15,16]=-inf",
         all(float(out[k]) == inf for k in (9, 10, 12, 13, 15, 16))),
    ]
    ok = all(c for _, c in checks)
    for name, c in checks:
        print(("PASS " if c else "FAIL ") + name)
    print("=> " + ("ALL PASS ✅ (規約B確定)" if ok else "FAILED ❌ (off-by-one 再発の疑い)"))
    return ok

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
