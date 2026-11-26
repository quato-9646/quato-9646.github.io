"""自分の牌譜を、学習済みモデルで1打ずつ評価する。

使い方:
    python review.py model_v5.pt 牌譜ファイル 対象 [閾値] [all]

    対象: プレイヤー名(一部でも可) または 席番号(0〜3)
    閾値: 自分の打牌の確率がこれ未満なら「要確認」として表示する(初期値 0.15)
    all : 最後に付けると、押し引きの場面を食い違いに関係なく全て表示する

出力は5つ:
  1. AIと違う打牌 … 自分の打牌の確率が低かった場面
  2. 押し引き   … 他家がリーチ中の場面で、AIと比べて「押しすぎ/降りすぎ」だった場面
  3. 手組み     … オーラスで満貫級の打点が必要なとき、打点を残せる打牌の提案
  4. リーチ判断 … 聴牌できる門前の手で、リーチ/ダマがAIと大きく違った場面(打点・点数状況を加味)
  5. 鳴き判断   … 鳴けた場面で、鳴く/スルーがAIと大きく違った場面(打点の見込み・点数状況を加味)
  (4と5は、model_v5.pt と同じフォルダに riichi_model.joblib と call_model.pt があれば出る)

mjcommon.py と rules.py と同じフォルダに置いて使う。
牌譜のファイル名が 2013052905gm-... のように年で始まっていれば、その年を使う
(それ以外は最近の年として扱う)。
"""
import math
import reasons
import os
import sys

try:
    import torch
except ImportError:
    torch = None

from mjcommon import (DEFAULT_YEAR, NAMES, DiscardNet, get_names, load, parse_game)
import review_decisions
import situation
from rules import LEVEL_NAMES, riichi_seats, seats_text, shanten_text

# 押し引きの判定の基準。「プロなら安全牌を切る確率」(降り確率)で分ける
AI_FOLD = 0.75   # これ以上なら、AIは降り寄り
AI_PUSH = 0.35   # これ以下なら、AIは押し寄り


def pick_seat(names, target):
    """「対象」の指定(席番号 or 名前の一部)から、席番号を決める。"""
    if target in ("0", "1", "2", "3"):
        return int(target)
    hits = [i for i, n in enumerate(names) if target in n]
    if len(hits) != 1:
        print("対象を1人に絞れませんでした。対局者:", names)
        print("名前の一部、または席番号(0〜3)で指定してください。")
        sys.exit(1)
    return hits[0]


def year_of(path):
    """ファイル名の先頭4桁を年として読む。読めなければ既定値。"""
    head = os.path.basename(path)[:4]
    return int(head) if head.isdigit() and 2000 <= int(head) <= 2100 else DEFAULT_YEAR


def tiles_text(counts):
    """34種の枚数 → '1m 2m 5p ...' の文字列。"""
    return " ".join(NAMES[t] for t in range(34) for _ in range(int(counts[t])))


def hand_text(x, draw):
    """手牌を表示用にする。ツモ牌は手牌から分けて表示する。"""
    c = x[0].astype(int).copy()
    tail = ""
    if draw is not None:
        c[draw] -= 1
        tail = f"   ツモ {NAMES[draw]}"
    return tiles_text(c) + tail


def push_fold(ev, probs, fold_p, pfm=None):
    """他家リーチ中の局面を、押し引きの観点で評価する。対象外ならNone。

    fold_p は「プロならこの場面で安全牌(現物・スジ・字牌の安全寄り)を切る確率」。
    自分が安全牌を切ったかどうかと比べて、押しすぎ/降りすぎを判定する。"""
    if ev["fold"] < 0:     # 押し引きの場面ではない(リーチ中の相手がいない/選択の余地なし/自分のリーチ宣言牌)
        return None
    x, a = ev["x"], ev["actual"]
    fold_model = fold_p
    ag = situation.aggression(ev["ctx"])           # 点数状況から見た、攻め(+1)〜守り(-1)
    learned = None
    level = {t: int(x[13][t]) - 1 for t in range(34) if x[0][t] > 0}   # 安全度の段階
    safe = [t for t, lv in level.items() if lv <= 2]
    risky = [t for t, lv in level.items() if lv > 2]
    user_safe = ev["fold"] == 1
    ai_side = "降り寄り" if fold_p >= AI_FOLD else "押し寄り" if fold_p <= AI_PUSH else "どちらとも言えない"

    # 代表の安全牌 = 切った後の向聴数が一番良い安全牌(同じならAIの確率が高い方)
    best_safe = min(safe, key=lambda t: (int(x[14][t]), -probs[t]))
    best_risky = max(risky, key=lambda t: probs[t])         # AIが押すなら切る牌
    if pfm is not None:       # 牌譜から学習した和了率・放銃率で、押す/降りるの価値を点数状況ごとに比べる
        shift, det_s, det_p, d_rel = review_decisions.pf_shift(ev, best_safe, best_risky, pfm)
        learned = {"safe": det_s, "push": det_p, "tiles": (best_safe, best_risky)}
    else:                     # 無ければ従来の経験則(攻め守りの度合いだけ)
        shift = 1.2 * ag["aggr"]
    fold_p = review_decisions.sig(review_decisions.logit(fold_model) - shift)
    ai_side = "降り寄り" if fold_p >= AI_FOLD else "押し寄り" if fold_p <= AI_PUSH else "どちらとも言えない"
    sh_actual = int(x[14][a]) - 1                           # 実際に切った後の向聴数
    sh_safe = int(x[14][best_safe]) - 1
    sh_risky = int(x[14][best_risky]) - 1

    verdict = "問題なし"
    if fold_p >= AI_FOLD and not user_safe:
        verdict = "押しすぎ候補"
    elif fold_p <= AI_PUSH and user_safe and sh_actual > sh_risky:
        verdict = "降りすぎ候補"     # 降りたせいで手が後退したときだけ。向聴数が同じなら降りて正解

    return {
        "seats": riichi_seats(ev["f"]), "level": level, "fold_p": fold_p, "ai_side": ai_side,
        "learned": learned, "fold_model": fold_model, "stance": situation.stance_text(ag), "aggr": ag["aggr"],
        "sit": situation.situation_report(ev["ctx"])[0][0],
        "user_safe": user_safe, "verdict": verdict,
        "best_safe": best_safe, "best_risky": best_risky,
        "sh_actual": sh_actual, "sh_safe": sh_safe, "sh_risky": sh_risky,
    }


def explain(r):
    """押し引き判定の理由を、短い文章にする。"""
    pf = r["pf"]
    a, lv = r["actual"], pf["level"][r["actual"]]
    text = f"プロの平均なら安全牌を切る確率 {pf['fold_model']:.0%}"
    if abs(pf["fold_p"] - pf["fold_model"]) >= 0.05:
        text += f" → 点数状況({pf['stance']})を加味して {pf['fold_p']:.0%}"
    text += (f" → AIは{pf['ai_side']}。"
            f"あなたは{LEVEL_NAMES[lv]}の{NAMES[a]}を切って{shanten_text(pf['sh_actual'])}。")
    if pf.get("learned"):
        lr, (ts_, tp_) = pf["learned"], pf["learned"]["tiles"]
        text += (f"牌譜から学習した見込み(この局): 安全牌の{NAMES[ts_]}を切ると 和了率{lr['safe']['win']:.0%}・放銃率{lr['safe']['dealin']:.0%}、"
                 f"危険牌の{NAMES[tp_]}を切ると 和了率{lr['push']['win']:.0%}・放銃率{lr['push']['dealin']:.0%}。")
    if pf["verdict"] == "押しすぎ候補":
        bs = pf["best_safe"]
        name = f"{NAMES[bs]}({LEVEL_NAMES[pf['level'][bs]]})"
        if pf["sh_safe"] <= pf["sh_actual"]:
            text += f"安全牌の{name}を切っても{shanten_text(pf['sh_safe'])}のままで、手を落とさずに降りられました。"
        else:
            text += (f"安全牌の{name}を切ると{shanten_text(pf['sh_safe'])}に後退します。")
            if pf["sh_actual"] >= 2:
                text += "手が遠いので、リーチに押す価値は低めです。"
            else:
                text += "手が近いので押す判断もありえますが、AIはより安全な切り方を好みます。"
    elif pf["verdict"] == "降りすぎ候補":
        br = pf["best_risky"]
        text += (f"AIは {NAMES[br]}({LEVEL_NAMES[pf['level'][br]]}) を切って"
                 f"{shanten_text(pf['sh_risky'])}を保つ押しを好みます。")
    return text


YAOCHU_T = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]


def style_potential(ev, t):
    """tを切った後の手の「打点の見込み」(目安)。ドラ・役牌・断幺九・染め手の見込みを足し合わせる。
    戻り値: (合計, 内訳 dict)"""
    x, ctx = ev["x"], ev["ctx"]
    hand = x[0].astype(int).copy()
    hand[t] -= 1
    melds = x[6].astype(int)
    tot = hand + melds
    n = max(int(tot.sum()), 1)
    dora = int((tot * x[5].astype(int)).sum())
    rw, sw = ctx["round_idx"] // 4, ctx["seat"]
    yk = 0.0
    for y in {31, 32, 33, 27 + rw, 27 + sw}:
        if melds[y] >= 3:
            yk += 2.0
        elif hand[y] >= 3:
            yk += 1.5
        elif hand[y] == 2:
            yk += 0.7
    tanyao = 1.0 if tot[YAOCHU_T].sum() == 0 else 0.0
    best = max(tot[s * 9:s * 9 + 9].sum() + tot[27:].sum() for s in range(3))
    flush = max(0.0, (best / n - 0.7) / 0.3) * 2.0
    parts = {"ドラ": float(dora), "役牌": yk, "断幺九": 0.8 * tanyao, "染め手": flush}
    return sum(parts.values()), parts


def style_advice(ev, probs, info):
    """オーラスで満貫級の打点が必要なとき、向聴数が同じ牌の中から、打点の見込みを残せる打牌を勧める。
    該当しなければ None。"""
    if info.get("policy") != "value":
        return None
    x = ev["x"]
    sh = {t: int(x[14][t]) - 1 for t in range(34) if x[0][t] > 0}
    best_sh = min(sh.values())
    cands = [t for t, v in sh.items() if v == best_sh]
    a = ev["actual"]
    if len(cands) < 2 or a not in cands or best_sh > 2:
        return None
    z = sum(probs[t] for t in cands) or 1.0
    pots = {t: style_potential(ev, t) for t in cands}
    score = {t: math.log(probs[t] / z + 1e-6) + 0.7 * pots[t][0] for t in cands}
    top = max(cands, key=lambda t: score[t])
    gain = pots[top][0] - pots[a][0]
    if top == a or gain < 1.0:
        return None
    diffs = [f"{k} {pots[top][1][k] - pots[a][1][k]:+.1f}" for k in pots[a][1]
             if abs(pots[top][1][k] - pots[a][1][k]) >= 0.3]
    return {"top": top, "gain": gain, "diffs": diffs, "sh": best_sh}


def review(rounds, model, me, pfm=None):
    """自分(席 me)の打牌ごとに、モデルの評価を記録する。"""
    rows = []
    skipped = 0
    for r in rounds:
        if r["broken"]:  # 手牌の追跡がずれた局は信用できないので評価しない
            skipped += 1
            continue
        for ev in r["events"]:
            if ev["who"] != me:
                continue
            with torch.no_grad():
                scores, fold_logit = model(torch.from_numpy(ev["x"])[None],
                                           torch.from_numpy(ev["f"])[None])
            probs = torch.softmax(scores, dim=1)[0]
            top = probs.topk(3)
            rows.append({
                "label": r["label"], "junme": ev["junme"],
                "hand": hand_text(ev["x"], ev["draw"]),
                "dora": tiles_text(ev["x"][5]),
                "actual": ev["actual"], "p": probs[ev["actual"]].item(),
                "top": [(int(i), float(p)) for p, i in zip(top.values, top.indices)],
                "riichi": ev["riichi"],
                "pf": push_fold(ev, probs.tolist(), torch.sigmoid(fold_logit).item(), pfm),
                "style": style_advice(ev, probs.tolist(), situation.situation_report(ev["ctx"])[1]),
                "ctx": ev["ctx"],
            })
            row = rows[-1]
            if row["actual"] != row["top"][0][0] and row["p"] < 0.15 and ev["x"][0][row["actual"]] > 0:
                row["why"], row["why_kind"] = reasons.compare(ev, row["top"][0][0], row["actual"], style_potential)
    return rows, skipped


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    model_path, log_path, target = sys.argv[1], sys.argv[2], sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "all" else 0.15
    show_all = sys.argv[-1] == "all"

    model = DiscardNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    root = load(log_path)
    names = get_names(root)
    me = pick_seat(names, target)
    print(f"評価対象: {names[me]} (席{me})   対局者: {' / '.join(names)}\n")

    rows, skipped = review(parse_game(root, year_of(log_path)), model, me,
                           review_decisions.load_pf_model(model_path))

    # ---- 1. AIと違う打牌 ----
    print(f"■ AIと違う打牌(自分の打牌の確率が {threshold:.0%} 未満)\n")
    flagged = 0
    for r in rows:
        if r["actual"] == r["top"][0][0] or r["p"] >= threshold:
            continue
        flagged += 1
        ai = ", ".join(f"{NAMES[t]}({p:.0%})" for t, p in r["top"])
        mark = " [リーチ宣言]" if r["riichi"] else ""
        print(f"{r['label']} {r['junme']}巡目{mark}   ドラ: {r['dora']}")
        print(f"  手牌: {r['hand']}")
        print(f"  自分: {NAMES[r['actual']]}({r['p']:.0%})  /  AI推奨: {ai}")
        if r.get("why"):
            print("  理由: " + "".join(r["why"]))
        print()
    if not flagged:
        print("該当なし\n")

    # ---- 2. 押し引き ----
    print("■ 押し引き(他家のリーチ中に、安全牌と危険牌の両方を切れた場面)\n")
    pf_rows = [r for r in rows if r["pf"]]
    shown = 0
    for r in pf_rows:
        pf = r["pf"]
        if not show_all and pf["verdict"] == "問題なし":
            continue
        shown += 1
        print(f"{r['label']} {r['junme']}巡目  [{seats_text(pf['seats'])}]   ドラ: {r['dora']}"
              f"   → {pf['verdict']}")
        print(f"  手牌: {r['hand']}")
        print(f"  状況: {pf['sit']}")
        print(f"  {explain(r)}\n")
    if not shown:
        print("該当なし\n")

    # ---- 3. 点数状況に応じた手組み ----
    print("■ 点数状況に応じた手組み(オーラスで満貫級の打点が必要な局面。向聴数が同じ打牌の中から)\n")
    st_rows = [r for r in rows if r["style"]]
    for r in st_rows:
        sa = r["style"]
        print(f"{r['label']} {r['junme']}巡目   ドラ: {r['dora']}")
        print(f"  手牌: {r['hand']}")
        print(f"  自分: {NAMES[r['actual']]} / 手組みの提案: {NAMES[sa['top']]}を切って、"
              f"打点の見込みを残す({', '.join(sa['diffs'])})。{shanten_text(sa['sh'])}は同じ")
        print(f"  状況: {situation.situation_report(r['ctx'])[0][0]}\n")
    if not st_rows:
        print("該当なし\n")

    # ---- 4. リーチ判断 / 5. 鳴き判断 ----
    rrows, crows = review_decisions.run(model_path, root, year_of(log_path), me, show_all)

    # ---- 要約 ----
    n = len(rows)
    agree = sum(r["actual"] == r["top"][0][0] for r in rows)
    in_top3 = sum(r["actual"] in [t for t, _ in r["top"]] for r in rows)
    print("=== 要約 ===")
    print(f"評価した打牌: {n} 回(リーチ後の自動打牌は除く)")
    print(f"AIの1位と一致: {agree / n:.1%}   AIの上位3位以内: {in_top3 / n:.1%}")
    print(f"要確認(確率 {threshold:.0%} 未満): {flagged} 回")
    k = len(pf_rows)
    if k:
        count = lambda key, v: sum(r["pf"][key] == v for r in pf_rows)
        print(f"押し引きの場面: {k} 回")
        print(f"  自分は 安全牌 {count('user_safe', True)} 回 / 危険牌 {count('user_safe', False)} 回")
        print(f"  AIは 降り寄り {count('ai_side', '降り寄り')} 回 / 押し寄り {count('ai_side', '押し寄り')} 回"
              f" / どちらとも言えない {count('ai_side', 'どちらとも言えない')} 回")
        print(f"  食い違い: 押しすぎ候補 {count('verdict', '押しすぎ候補')} 回"
              f" / 降りすぎ候補 {count('verdict', '降りすぎ候補')} 回")
    else:
        print("押し引きの場面: なし")
    if rrows is not None or crows is not None:
        review_decisions.summary(rrows, crows)
    if skipped:
        print(f"※ 手牌の追跡が合わなかった {skipped} 局は評価から除きました")


if __name__ == "__main__":
    main()
