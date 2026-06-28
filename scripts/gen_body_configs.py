"""本体12本のconfig生成。ベース blt_cblt_gate2_ja.yaml を流用し patching等のparamのみ変更。
12本 = byte-BLT×5rate + CBLT-√ℓ×5rate + Sum(r6) + Avg(r6)。両アーム monotonic。
共通: largecorpus全(full epoch 561k steps @ b8×s8192=36.75B tok)、realtime_patching + 凍結285k、warmup500。
各config: patching_mode/cblt_aggregation/threshold/patch_size のみ差し替え(他は完全同一)。"""
import yaml, copy, os

BASE = "/gs/bs/tga-RLA/yoshida/blt/bytelatent/configs/blt_cblt_gate2_ja.yaml"
OUT = "/gs/bs/tga-RLA/yoshida/blt/bytelatent/configs"
LC = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus"
FROZEN = "/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated"
FULL_EPOCH_STEPS = 1121000  # 36.75B tok / (b8×s4096=32768)。seq_len は max_seqlen 4096 以下必須

# 本データ較正θ(nats)
BYTE_TH = {4: 0.3008, 5: 0.5000, 6: 0.6367, 7: 0.7344, 8: 0.8516}
CBLT_TH = {4: -0.0035, 5: 0.0067, 6: 0.0937, 7: 0.2439, 8: 0.3934}
SUM_TH6, AVG_TH6 = 0.1123, 0.0723

# (name, patch_size/rate, model_patching_mode, cblt_aggregation, threshold)
SPECS = []
for r in [4, 5, 6, 7, 8]:
    SPECS.append((f"blt_body_byte_r{r}", r, "entropy", "sqrt", BYTE_TH[r]))   # byte-BLT: entropy mode (agg無視)
    SPECS.append((f"blt_body_cblt_r{r}", r, "cblt", "sqrt", CBLT_TH[r]))      # CBLT-√ℓ
SPECS.append(("blt_body_sum_r6", 6, "cblt", "sum", SUM_TH6))
SPECS.append(("blt_body_avg_r6", 6, "cblt", "avg", AVG_TH6))

base = yaml.safe_load(open(BASE))

def make(name, rate, pmode, agg, th):
    c = copy.deepcopy(base)
    c["name"] = name
    c["dump_dir"] = f"/gs/bs/tga-RLA/yoshida/blt_runs/body/{name}"
    c["steps"] = FULL_EPOCH_STEPS
    c["seed"] = 777
    c["train_entropy_model"] = False
    # optim: warmup 500 (smoke=10), lr 4e-4 維持
    c["optim"]["warmup"] = 500
    c["optim"]["lr"] = 4e-04
    c["optim"]["lr_min_ratio"] = 0.1
    # model: patching 差し替え + monotonicity
    c["model"]["patch_size"] = float(rate)
    c["model"]["patching_mode"] = pmode
    c["model"]["patching_threshold"] = th
    c["model"]["monotonicity"] = True
    c["model"]["max_seqlen"] = 4096
    c["model"]["max_encoder_seq_length"] = 4096
    # data: largecorpus + realtime_patching + 凍結285k
    c["data"]["root_dir"] = LC
    c["data"]["preprocess_dir"] = f"{LC}/preprocess"
    c["data"]["sources"] = {"corpus": 1.0}
    c["data"]["batch_size"] = 8
    c["data"]["seq_len"] = 4096   # max_seqlen 4096 以下必須(超過でCUDA illegal access)
    c["data"]["max_encoder_seq_length"] = 4096
    c["data"]["add_patches"] = True
    c["data"]["patcher_args"] = {
        "patching_mode": pmode,
        "cblt_aggregation": agg,
        "threshold": th,
        "monotonicity": True,
        "patching_device": "cuda",
        "realtime_patching": True,
        "entropy_model_checkpoint_dir": FROZEN,
    }
    # checkpoint: 収束カーブ用に刻む
    c["checkpoint"]["dump"]["every"] = 2000
    c["checkpoint"]["dump"]["keep"] = -1
    path = f"{OUT}/{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return path

for name, rate, pmode, agg, th in SPECS:
    p = make(name, rate, pmode, agg, th)
    print(f"[OK] {name}: mode={pmode} agg={agg if pmode=='cblt' else '-'} rate={rate} θ={th}")
print(f"\n計 {len(SPECS)} 本生成 → {OUT}/blt_body_*.yaml")
