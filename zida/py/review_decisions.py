"""リーチ判断・鳴き判断の評価と、理由の文章づくり(review.py から呼ばれる)。

  riichi_model.joblib … 聴牌できる手で「リーチかダマか」(決定木。打点の見積もりも使う)
  outcome_model.joblib … リーチ/ダマの和了率・放銃率(牌譜の結果から学習。無ければ経験則で代用)
  call_model.pt       … 他家の打牌に「パス/チー/ポン/大明槓」(ニューラルネット)
どちらも train_decisions.py で作る。review.py に渡したモデルと同じフォルダに置く。

プロの平均的な判断(モデル)に、次の2つの「状況補正」を重ねて総合判断にしている。
  1. 打点: 手の役・符・翻を計算し、ダマとリーチで和了したときの点数を比べる
  2. 点数状況: 順位・点差・残り局数から、攻めるべきか守るべきか、何点の手が要るかを求める(situation.py)
"""
import math
import os

import numpy as np
try:
    import torch
except ImportError:
    torch = None

import situation
from decisions import CALL_NAMES, DecisionNet, call_outcome_features, parse_decisions, pf_features, riichi_features
from mjcommon import NAMES
from rules import shanten_text
from scoring import describe, payments

# 判定の基準(状況補正後の確率)。プロの検証用データで、該当する割合が数%以下になる値。
RIICHI_DAMA_BELOW = 0.15    # リーチ確率がこれ未満 → ダマ推奨
RIICHI_GO_ABOVE = 0.85      # リーチ確率がこれより上 → リーチ推奨
CALL_GO_ABOVE = 0.80        # 鳴く確率がこれより上 → 鳴き推奨
CALL_NO_BELOW = 0.10        # 鳴く確率がこれ未満 → 鳴かない推奨

SEAT_REL = {1: "下家", 2: "対面", 3: "上家"}
W_RON, W_TSUMO = 0.6, 0.4   # 和了のうちロン・ツモの割合の目安
DAMA_WIN_BONUS = 1.1        # ダマは相手が降りないぶん、同じ待ちでもリーチより和了しやすいとする倍率


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def pts(v):
    return f"{int(round(v)):,}"


def tiles_text(counts):
    return " ".join(NAMES[t] for t in range(34) for _ in range(int(counts[t])))


def load_models(model_path):
    """review.py のモデルと同じフォルダから、リーチ/鳴きのモデルを探して読む。無ければNone。"""
    folder = os.path.dirname(os.path.abspath(model_path))
    riichi = call = om = cm = None
    rp, cp = os.path.join(folder, "riichi_model.joblib"), os.path.join(folder, "call_model.pt")
    op = os.path.join(folder, "outcome_model.joblib")
    cop = os.path.join(folder, "call_outcome_model.joblib")
    if os.path.exists(cop):
        try:
            import joblib
            cm = joblib.load(cop)
        except Exception as e:
            print(f"※ 鳴きの和了率・放銃率モデルを読めませんでした({e!r})。経験則で代用します")
    if os.path.exists(op):
        try:
            import joblib
            om = joblib.load(op)
        except Exception as e:
            print(f"※ 和了率・放銃率モデルを読めませんでした({e!r})。経験則で代用します")
    if os.path.exists(rp):
        try:
            import joblib
            riichi = joblib.load(rp)
        except Exception as e:      # scikit-learn が無い等
            print(f"※ リーチ判断モデルを読めませんでした({e!r})")
    if os.path.exists(cp):
        call = DecisionNet(4)
        try:
            call.load_state_dict(torch.load(cp, map_location="cpu"))
            call.eval()
        except Exception as e:
            print(f"※ 鳴き判断モデルを読めませんでした({e!r})。train_decisions.py で作り直してください")
            call = None
    return riichi, call, om, cm


def situation_lines(ctx, with_need=True):
    """状況の説明を1〜3行で。"""
    lines, info = situation.situation_report(ctx)
    return lines, info


# ---------------------------------------------------------------
# リーチ / ダマ
# ---------------------------------------------------------------
def base_materials(ev):
    """プロ平均のリーチ判断(待ち・巡目・他家・順位など)の材料。(リーチ寄り, ダマ寄り)"""
    f, g = ev["f"], ev["g"]
    live, junme = ev["live"], ev["junme"]
    others = int(round(float(f[1:4].sum())))
    rank, dora = int(g[13]), int(g[9])
    go, dama = [], []
    if live >= 7:
        go.append(f"待ちが広い(残り{live}枚)")
    elif live <= 4:
        dama.append(f"待ちが薄い(残り{live}枚)")
    if junme <= 8:
        go.append(f"{junme}巡目と早く、和了までの時間がある")
    elif junme >= 13:
        dama.append(f"{junme}巡目と遅く、リーチしても和了しにくい")
    if others >= 1:
        dama.append(f"他家のリーチが{others}人いて、リーチ後は降りられない")
    if rank >= 3:
        go.append(f"{rank}着目で、打点を上げたい")
    elif rank == 1:
        dama.append("トップ目で、無理に攻める必要が小さい")
    return go, dama


def value_gains(ctx, tv):
    """ダマとリーチのそれぞれで和了したときの、順位期待値の増え。(ダマ, リーチ, 詳細)"""
    d, r = tv["dama"], tv["riichi"]
    yp = d["yaku_p"]
    freq_d = W_RON * yp + W_TSUMO
    g_d_ron = situation.win_gain(ctx, d["ron_if"], False) if yp > 0 else 0.0
    g_d_tsu = situation.win_gain(ctx, d["tsumo"], True)
    gain_d = (W_RON * yp * g_d_ron + W_TSUMO * g_d_tsu) / freq_d
    gain_r = (W_RON * situation.win_gain(ctx, r["ron_x"], False, riichi=True)
              + W_TSUMO * situation.win_gain(ctx, r["tsumo_x"], True, riichi=True))
    return gain_d, gain_r, freq_d


def equity_pair(ctx, tv, om=None, A=None):
    """リーチとダマの「この局の順位期待値の増え」(eg_r, eg_d)と、その内訳を返す。
    om(和了率・放銃率のモデル)があれば牌譜から学習した確率を使い、無ければ経験則を使う。"""
    d, r = tv["dama"], tv["riichi"]
    if om is None or A is None:
        gain_d, gain_r, freq_d = value_gains(ctx, tv)
        return gain_r, freq_d * DAMA_WIN_BONUS * gain_d, None
    yp = d["yaku_p"]
    out, det = {}, {}
    for flag in (0, 1):
        X = np.concatenate([A, np.full((1, 1), flag, dtype=np.float32)], 1)
        pw = float(om["win"].predict_proba(X)[0, 1])
        pd_ = float(om["dealin"].predict_proba(X)[0, 1])
        ts = float(om["tsumo"].predict_proba(X)[0, 1])
        if flag == 0:
            ron_pts, tsu_pts = d["ron_if"], d["tsumo"]
            if yp <= 0:
                ts = 1.0                       # 役がないダマは、ツモでしか和了できない
        else:
            ron_pts, tsu_pts = r["ron_x"], r["tsumo_x"]
        g_tsu = situation.win_gain(ctx, tsu_pts, True, riichi=bool(flag))
        g_ron = situation.win_gain(ctx, ron_pts, False, riichi=bool(flag)) if ts < 1.0 else 0.0
        gain = ts * g_tsu + (1 - ts) * g_ron
        loss = situation.loss_cost(ctx, om["loss"][flag])
        stick = situation.loss_cost(ctx, 1000) * max(1 - pw - pd_, 0.0) if flag else 0.0
        out[flag] = pw * gain - pd_ * loss - stick
        det[flag] = {"win": pw, "dealin": pd_, "tsumo": ts}
    return out[1], out[0], det


def riichi_situation(ev, tv, om=None, A=None):
    """リーチ/ダマの状況補正。戻り値: dict(shift, lines, ...)"""
    ctx = ev["ctx"]
    sit_lines, info = situation_lines(ctx)
    out = {"shift": 0.0, "lines": [], "sit": sit_lines}
    if not tv or not tv.get("riichi"):
        return out
    d, r = tv["dama"], tv["riichi"]
    eg_r, eg_d, det = equity_pair(ctx, tv, om, A)
    diff = eg_r - eg_d
    # 打点そのものの差(役なしのダマなど)は、モデルがすでに織り込んでいる。
    # 状況補正は「普通の状況(全員同点・序盤)と比べて、打点の価値がどれだけ変わるか」だけを使う。
    neutral = dict(ctx, round_idx=0, last_round=7, honba=0, sticks=0, scores=[25000] * 4)
    n_r, n_d, _ = equity_pair(neutral, tv, om, A)
    diff_rel = diff - (n_r - n_d)
    shift_value = 0.8 * math.tanh(diff_rel / 8.0)
    shift_aggr = 0.4 * info["aggr"]
    out["shift"] = shift_value + shift_aggr
    out["diff"], out["aggr"] = diff, info["aggr"]

    # 打点の説明
    if d["yaku_p"] <= 0:
        dama_txt = f"ダマ: ロン不可(役なし)・ツモ平均{pts(d['tsumo'])}点"
    else:
        dama_txt = f"ダマ: ロン平均{pts(d['ron_if'])}点・ツモ平均{pts(d['tsumo'])}点"
        if d["yaku_p"] < 0.999:
            dama_txt += f"(待ちの{d['yaku_p']:.0%}は役あり)"
    w0 = max((w for w in tv["waits"] if w["ron"]), key=lambda w: w["live"], default=None)
    if w0:
        dama_txt += f"。例: {NAMES[w0['tile']]}で " + describe(w0["ron"])
    out["lines"].append(dama_txt)
    out["lines"].append(f"リーチ(裏ドラ・一発の期待込み): ロン平均{pts(r['ron_x'])}点・ツモ平均{pts(r['tsumo_x'])}点")
    if det:
        out["lines"].append(f"牌譜から学習した見込み(この局): 和了率 ダマ{det[0]['win']:.0%} / リーチ{det[1]['win']:.0%}、"
                            f"放銃率 ダマ{det[0]['dealin']:.0%} / リーチ{det[1]['dealin']:.0%}")

    # オーラスの必要打点との比較
    need = info.get("need")
    if need and need["ron_min"]:
        d_ok = d["yaku_p"] > 0 and d["ron_if"] >= need["ron_min"]
        r_ok = r["ron_x"] >= need["ron_min"]
        mark = lambda ok: "届く" if ok else "届かない"
        out["lines"].append(f"順位が上がるロンは最低{pts(need['ron_min'])}点以上(相手により更に必要) → ダマは{mark(d_ok)} / リーチは{mark(r_ok)}")
        if r_ok and not d_ok:
            out["lines"].append("→ この局面は、リーチで打点を上げないと順位が変わらない")
        elif d_ok:
            out["lines"].append("→ ダマの打点で足りるので、待ちを狭めず・相手を降ろさないダマが合理的")
    elif diff_rel >= 4:
        out["lines"].append("→ この順位・点差では、リーチで打点を上げる価値が普段より大きい")
    elif diff_rel <= -4:
        out["lines"].append("→ この順位・点差では、リーチで打点を上げる価値が普段より小さい(ダマで十分)")
    return out


def cut_comparison(ev):
    """聴牌になる切り方ごとの、待ち・打点・和了時の順位期待値を比べる(上位を返す)。"""
    ctx, rows = ev["ctx"], []
    for t, tv in ev.get("values", {}).items():
        if not tv or not tv.get("riichi"):
            continue
        live = sum(w["live"] for w in tv["waits"])
        gain_d, gain_r, freq_d = value_gains(ctx, tv)
        pwin = 0.62 * (1 - math.exp(-0.28 * live))
        best_r = pwin * gain_r
        best_d = pwin * freq_d * DAMA_WIN_BONUS * gain_d
        rows.append({"t": t, "live": live, "waits": [w["tile"] for w in tv["waits"]],
                     "ev": max(best_r, best_d), "mode": "リーチ" if best_r >= best_d else "ダマ",
                     "dama": tv["dama"]["ron_if"] if tv["dama"]["yaku_p"] > 0 else 0,
                     "riichi": tv["riichi"]["ron_x"]})
    rows.sort(key=lambda r: -r["ev"])
    return rows


def explain_riichi(r):
    ev, p, sit = r["ev"], r["p_model"], r["sit"]
    go, dama = base_materials(ev)
    tv = r["tv"]
    waits = " ".join(NAMES[t] for t in ev["waits"])
    head = (f"{NAMES[ev['best_t']]}切りの待ちは {waits}(残り{ev['live']}枚)。"
            f"プロの平均ならリーチする確率 {p:.0%}。")
    mats = go if p >= 0.5 else dama
    if mats:
        head += ("リーチ" if p >= 0.5 else "ダマ") + "向きの材料: " + " / ".join(mats) + "。"
    lines = [head] + sit["lines"]
    if abs(sit["shift"]) >= 0.3:
        lines.append(f"点数状況・打点を加味した総合のリーチ確率: {r['p']:.0%}")
    cmp = r["cmp"]
    if cmp and len(cmp) >= 2 and ev.get("actual_t") is not None:
        best = cmp[0]
        mine = next((c for c in cmp if c["t"] == ev["actual_t"]), None)
        if mine and best["t"] != mine["t"] and best["ev"] - mine["ev"] >= 3.0:
            lines.append(f"切り方の比較: {NAMES[best['t']]}切り(待ち{best['live']}枚・{best['mode']}想定)の方が"
                         f"和了時の順位期待値を含めた見込みが高い(あなたは{NAMES[mine['t']]}切り)")
    return lines


def judge_riichi(riichi_model, rounds, me, om=None):
    rows = []
    for r in rounds:
        if r["broken"]:
            continue
        for ev in r["riichi_events"]:
            if ev["who"] != me or not ev["eligible"]:
                continue
            if ev["label"] == 0 and not ev.get("kept", 1):
                continue        # 聴牌を取らなかった打牌は「リーチかダマか」の判断ではないので除く
            A = riichi_features(ev["x"][None], ev["f"][None], ev["g"][None], ev["V"][None])
            p_model = float(riichi_model.predict_proba(A)[0, 1])
            t_used = ev.get("actual_t") if ev.get("values", {}).get(ev.get("actual_t")) else ev["best_t"]
            tv = ev.get("values", {}).get(t_used)
            sit = riichi_situation(ev, tv, om, A)
            p = sig(logit(p_model) + sit["shift"])
            verdict = "問題なし"
            if ev["label"] == 1 and p < RIICHI_DAMA_BELOW:
                verdict = "ダマ推奨なのにリーチ"
            elif ev["label"] == 0 and p > RIICHI_GO_ABOVE:
                verdict = "リーチ推奨なのにダマ"
            rows.append({"round": r["label"], "ev": ev, "p_model": p_model, "p": p, "label": ev["label"],
                         "verdict": verdict, "sit": sit, "tv": tv, "cmp": cut_comparison(ev)})
    return rows


# ---------------------------------------------------------------
# 鳴き
# ---------------------------------------------------------------
def call_outlook(ev, kind):
    """鳴いた後の手の見込み(目安)。役が確定しているか・見込み打点はどれくらいか。"""
    g = ev["g"]
    yaku_han = int(g[24]) + (int(g[20]) if kind in (2, 3) else 0)
    tanyao = bool(g[31]) if kind in (1, 2, 3) else False
    flush_han = 2 if g[26] == 0 and g[27] >= 1 else (1 if g[26] <= 2 and g[27] >= 1 else 0)
    dora = int(g[29]) + int(g[21])
    sure = yaku_han > 0
    possible = sure or tanyao or flush_han > 0
    est_han = yaku_han + (1 if tanyao else 0) + flush_han + dora
    dealer = ev["ctx"]["seat"] == 0
    est_pts = payments(max(est_han, 1), 30, dealer)["ron"] if possible else 0
    return {"sure": sure, "possible": possible, "est_han": est_han, "est_pts": est_pts,
            "yaku_han": yaku_han, "tanyao": tanyao, "dora": dora, "flush_han": flush_han}


def call_reasons(ev, kind):
    """kind(1=チー 2=ポン 3=大明槓)で鳴く/鳴かないの材料。(鳴く寄り, スルー寄り)"""
    g, f = ev["g"], ev["f"]
    now, after = ev["sh_now"], ev["sh_after"][kind]
    others = int(round(float(f[1:4].sum())))
    opened, junme = int(g[11]), ev["junme"]
    ol = call_outlook(ev, kind)
    go, no = [], []
    if after is not None:
        if after < 0:
            go.append(f"{shanten_text(now)}から和了形に届く")
        elif after < now:
            go.append(f"{shanten_text(now)}から{shanten_text(after)}に進む")
        elif after == now:
            no.append(f"{shanten_text(now)}のままで、手が進まない")
        else:
            no.append(f"{shanten_text(now)}から{shanten_text(after)}に後退する")
    if ol["yaku_han"]:
        go.append("役牌があり、鳴いても役が確定している")
    elif ol["tanyao"]:
        go.append("全部が中張牌で、断幺九の役が付く")
    elif ol["flush_han"]:
        go.append("染め手(混一色)の形に近く、役が付く見込み")
    else:
        no.append("鳴いても役が確定せず、役牌・断幺九・染め手の見込みも薄い(和了できない恐れ)")
    if ol["dora"]:
        go.append(f"ドラが{ol['dora']}枚あり、打点が見込める")
    if opened == 0 and now <= 1:
        no.append("門前で聴牌が近く、鳴くとリーチ・打点を捨てる")
    if others >= 1:
        no.append(f"他家のリーチが{others}人いて、鳴くと降りづらくなる")
    if junme >= 13:
        no.append("巡目が遅く、鳴いても間に合いにくい")
    elif junme <= 6 and after is not None and after < now:
        go.append("巡目が早く、速度を上げる価値が大きい")
    return go, no, ol


def call_equity(ctx, ev, kind, cm):
    """鳴いた場合とパスの場合の、この局の順位期待値の増え。(鳴き, パス, 内訳)"""
    out, det = {}, {}
    for a in (kind, 0):
        A = call_outcome_features(ev["x"][None], ev["f"][None], ev["g"][None], [a])
        pw = float(cm["win"].predict_proba(A)[0, 1])
        pd_ = float(cm["dealin"].predict_proba(A)[0, 1])
        win_pts = max(float(cm["pts"].predict(A)[0]), 1000.0)
        ts = cm["tsumo_share"]
        gain = ts * situation.win_gain(ctx, win_pts, True) + (1 - ts) * situation.win_gain(ctx, win_pts, False)
        out[a] = pw * gain - pd_ * situation.loss_cost(ctx, cm["loss"])
        det[a] = {"win": pw, "dealin": pd_, "pts": win_pts}
    return out[kind], out[0], det


def call_situation(ev, kind, cm=None):
    """鳴きの状況補正。戻り値 dict(shift, lines)"""
    ctx = ev["ctx"]
    sit_lines, info = situation_lines(ctx)
    ol = call_outlook(ev, kind)
    after, now = ev["sh_after"][kind], ev["sh_now"]
    if cm is not None:
        eg_c, eg_p, det = call_equity(ctx, ev, kind, cm)
        neutral = dict(ctx, round_idx=0, last_round=7, honba=0, sticks=0, scores=[25000] * 4)
        n_c, n_p, _ = call_equity(neutral, ev, kind, cm)
        diff_rel = (eg_c - eg_p) - (n_c - n_p)
        shift = 0.8 * math.tanh(diff_rel / 8.0) + 0.4 * info["aggr"]
        lines = [f"牌譜から学習した見込み(この局): 和了率 パス{det[0]['win']:.0%} → {CALL_NAMES[kind]}{det[kind]['win']:.0%}、"
                 f"放銃率 パス{det[0]['dealin']:.0%} → {CALL_NAMES[kind]}{det[kind]['dealin']:.0%}、"
                 f"和了時の予測 約{pts(det[kind]['pts'])}点"]
        if diff_rel >= 4:
            lines.append("→ この順位・点差では、鳴いて和了率を上げる価値が普段より大きい")
        elif diff_rel <= -4:
            lines.append("→ この順位・点差では、鳴く価値が普段より小さい")
        return {"shift": shift, "lines": sit_lines + lines, "info": info}
    shift, lines = 0.5 * info["aggr"], []
    if not ol["possible"]:
        shift -= 0.8
    need = info.get("need")
    if need and need["ron_min"] and ol["possible"]:
        if ol["est_pts"] < need["ron_min"] * 0.9:
            shift -= 1.2
            lines.append(f"この鳴きの見込み打点は約{pts(ol['est_pts'])}点で、順位が上がる目安"
                         f"{pts(need['ron_min'])}点に届きにくい")
        elif after is not None and after < now:
            shift += 0.4
            lines.append(f"見込み打点(約{pts(ol['est_pts'])}点)が必要点({pts(need['ron_min'])}点)を満たし、手も進む")
    elif need and need["ron_min"] and not ol["possible"]:
        lines.append("役の見込みが薄く、和了できなければ順位は変わらない")
    return {"shift": shift, "lines": sit_lines + lines, "info": info}


def explain_call(r):
    ev, pr = r["ev"], r["probs"]
    t = ev["tile"]
    rel = SEAT_REL.get(int(ev["g"][23]), "他家")
    legal = [CALL_NAMES[k] for k in (1, 2, 3) if ev["legal"][k]]
    head = (f"{rel}の{NAMES[t]}を {'・'.join(legal)}できる場面。"
            f"プロの平均の選択確率: " + " / ".join(f"{CALL_NAMES[k]} {pr[k]:.0%}" for k in range(4) if ev["legal"][k]) + "。")
    kind = ev["label"] if (r["verdict"] == "鳴かない推奨なのに鳴いた") else r["best_call"]
    go, no, ol = call_reasons(ev, kind)
    if r["verdict"] == "鳴き推奨なのにスルー" or (r["verdict"] == "問題なし" and r["p_call"] >= 0.5):
        mats, side = go, f"{CALL_NAMES[kind]}向きの材料: "
    else:
        mats, side = no, "スルー向きの材料: "
    lines = [head + (side + " / ".join(mats) + "。" if mats else "")]
    lines += r["sit"]["lines"]
    if abs(r["sit"]["shift"]) >= 0.3:
        lines.append(f"点数状況を加味した総合の鳴く確率: {r['p_call_adj']:.0%}")
    return lines


def judge_call(call_model, rounds, me, cm=None):
    rows = []
    for r in rounds:
        if r["broken"]:
            continue
        for ev in r["call_events"]:
            if ev["p"] != me:
                continue
            if hasattr(call_model, "infer"):      # numpy版(torch無し)
                pr = call_model.infer(ev["x"], ev["f"], ev["g"], ev["legal"])
            else:
                with torch.no_grad():
                    logit_ = call_model(torch.from_numpy(ev["x"])[None], torch.from_numpy(ev["f"])[None],
                                        torch.from_numpy(ev["g"])[None], torch.tensor([ev["legal"]]))
                pr = torch.softmax(logit_, 1)[0].tolist()
            p_call = 1 - pr[0]
            best_call = max((k for k in (1, 2, 3) if ev["legal"][k]), key=lambda k: pr[k])
            sit = call_situation(ev, best_call, cm)
            p_adj = sig(logit(p_call) + sit["shift"])
            verdict = "問題なし"
            if ev["label"] == 0 and p_adj > CALL_GO_ABOVE:
                verdict = "鳴き推奨なのにスルー"
            elif ev["label"] > 0 and p_adj < CALL_NO_BELOW:
                verdict = "鳴かない推奨なのに鳴いた"
            rows.append({"round": r["label"], "ev": ev, "probs": pr, "p_call": p_call, "p_call_adj": p_adj,
                         "best_call": best_call, "verdict": verdict, "sit": sit})
    return rows


# ---------------------------------------------------------------
# 表示
# ---------------------------------------------------------------
def print_riichi(rows, show_all):
    print("\n■ リーチ判断(聴牌できる門前の手で、リーチかダマか。打点と点数状況を加味)\n")
    shown = 0
    for r in rows:
        if not show_all and r["verdict"] == "問題なし":
            continue
        shown += 1
        ev = r["ev"]
        mine = "リーチ" if r["label"] else "ダマ"
        print(f"{r['round']} {ev['junme']}巡目   ドラ: {tiles_text(ev['x'][5])}   自分: {mine}   → {r['verdict']}")
        print(f"  手牌: {tiles_text(ev['x'][0])}")
        for line in explain_riichi(r):
            print(f"  {line}")
        print()
    if not shown:
        print("該当なし\n")


def print_call(rows, show_all):
    print("\n■ 鳴き判断(他家の打牌に、鳴けた場面。打点の見込みと点数状況を加味)\n")
    shown = 0
    for r in rows:
        if not show_all and r["verdict"] == "問題なし":
            continue
        if show_all and r["p_call_adj"] < 0.05 and r["ev"]["label"] == 0:
            continue        # 明らかにスルーの場面は、all でも省く(数が多すぎるため)
        shown += 1
        ev = r["ev"]
        mine = CALL_NAMES[ev["label"]]
        print(f"{r['round']} {ev['junme']}巡目   ドラ: {tiles_text(ev['x'][5])}   自分: {mine}   → {r['verdict']}")
        print(f"  手牌: {tiles_text(ev['x'][0])}")
        for line in explain_call(r):
            print(f"  {line}")
        print()
    if not shown:
        print("該当なし\n")


def summary(rrows, crows):
    if rrows is not None:
        n = len(rrows)
        print(f"リーチ判断の場面: {n} 回")
        if n:
            agree = sum((r["p"] >= 0.5) == bool(r["label"]) for r in rrows)
            print(f"  自分: リーチ {sum(r['label'] for r in rrows)} 回 / ダマ {n - sum(r['label'] for r in rrows)} 回"
                  f"   AI(状況補正後)と同じ側: {agree / n:.0%}")
            print(f"  食い違い: ダマ推奨なのにリーチ {sum(r['verdict'] == 'ダマ推奨なのにリーチ' for r in rrows)} 回"
                  f" / リーチ推奨なのにダマ {sum(r['verdict'] == 'リーチ推奨なのにダマ' for r in rrows)} 回")
    if crows is not None:
        n = len(crows)
        called = sum(r["ev"]["label"] > 0 for r in crows)
        print(f"鳴ける場面: {n} 回(実際に鳴いた {called} 回)")
        if n:
            print(f"  食い違い: 鳴き推奨なのにスルー {sum(r['verdict'] == '鳴き推奨なのにスルー' for r in crows)} 回"
                  f" / 鳴かない推奨なのに鳴いた {sum(r['verdict'] == '鳴かない推奨なのに鳴いた' for r in crows)} 回")


def run(model_path, root, year, me, show_all):
    """リーチ判断・鳴き判断を評価して表示し、要約用の行を返す。モデルが無ければ案内だけ出す。"""
    riichi_model, call_model, om, cm = load_models(model_path)
    if riichi_model is None and call_model is None:
        print("\n※ riichi_model.joblib / call_model.pt が見つからないため、リーチ判断・鳴き判断は省略しました。")
        return None, None
    rounds = parse_decisions(root, year, full=True)
    rrows = crows = None
    if riichi_model is not None:
        rrows = judge_riichi(riichi_model, rounds, me, om)
        print_riichi(rrows, show_all)
    if call_model is not None:
        crows = judge_call(call_model, rounds, me, cm)
        print_call(crows, show_all)
    return rrows, crows


# ---------------------------------------------------------------
# 押し引き(他家のリーチ中)
# ---------------------------------------------------------------
def load_pf_model(model_path):
    """押し引きの和了率・放銃率モデル(pf_outcome_model.joblib)。無ければ None。"""
    path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "pf_outcome_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        print(f"※ 押し引きの和了率・放銃率モデルを読めませんでした({e!r})。経験則で代用します")
        return None


def pf_equity(ev, ctx, tile, pfm):
    """tile を切った場合の、この局の順位期待値の増えと、和了率・放銃率・和了時の点数。"""
    A = pf_features(ev, tile)[None]
    pw = float(pfm["win"].predict_proba(A)[0, 1])
    pd_ = float(pfm["dealin"].predict_proba(A)[0, 1])
    win_pts = max(float(pfm["pts"].predict(A)[0]), 1000.0)
    ts = pfm["tsumo_share"]
    gain = ts * situation.win_gain(ctx, win_pts, True) + (1 - ts) * situation.win_gain(ctx, win_pts, False)
    eg = pw * gain - pd_ * situation.loss_cost(ctx, pfm["loss"])
    return eg, {"win": pw, "dealin": pd_, "pts": win_pts}


def pf_shift(ev, safe_tile, push_tile, pfm):
    """押す方向の補正(logit)と、安全牌/危険牌それぞれの見込み。点数状況から見て、押す価値が普段より大きければ正。"""
    ctx = ev["ctx"]
    info = situation.situation_report(ctx)[1]
    eg_s, det_s = pf_equity(ev, ctx, safe_tile, pfm)
    eg_p, det_p = pf_equity(ev, ctx, push_tile, pfm)
    neutral = dict(ctx, round_idx=0, last_round=7, honba=0, sticks=0, scores=[25000] * 4)
    n_s, _ = pf_equity(ev, neutral, safe_tile, pfm)
    n_p, _ = pf_equity(ev, neutral, push_tile, pfm)
    diff_rel = (eg_p - eg_s) - (n_p - n_s)
    shift = 0.8 * math.tanh(diff_rel / 8.0) + 0.4 * info["aggr"]
    return shift, det_s, det_p, diff_rel
