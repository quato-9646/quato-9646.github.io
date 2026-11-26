"""牌譜を1打ずつ送りながら、AIの推奨牌とコメントを見られるHTMLを作る。

使い方:
    python make_viewer.py model_v5.pt 出力.html 対象 牌譜1 [牌譜2 ...]

    対象: プレイヤー名(一部でも可) または 席番号(0〜3)
できた HTML は1ファイルで完結していて、ブラウザで開くだけで使える(サーバー不要)。
    ← → キー(スマホは ◀ ▶ ボタン)で1打ずつ進む。「確認」ボタンでAIと違う場面へ飛べる。
    対象はまず表示する視点で、画面上のプレイヤー名を押すと他の3人の視点に切り替わる。
"""
import base64
import json
import math
import os
import sys

try:
    import torch
except ImportError:
    torch = None

import reasons
import review
import review_decisions
import reads
import situation
import tactics
from decisions import parse_decisions
from mjcommon import build_meld_list, AKA_IDS, DEFAULT_YEAR, NAMES, DiscardNet, decode_meld, get_names, load, parse_game
from rules import LEVEL_NAMES, shanten_text

SEAT_REL = ["自分", "下家", "対面", "上家"]


def expand(counts):
    return [t for t in range(34) for _ in range(int(counts[t]))]


YAKU = ["門前清自摸和", "立直", "一発", "槍槓", "嶺上開花", "海底摸月", "河底撈魚", "平和", "断幺九", "一盃口",
        "自風 東", "自風 南", "自風 西", "自風 北", "場風 東", "場風 南", "場風 西", "場風 北",
        "役牌 白", "役牌 發", "役牌 中", "両立直", "七対子", "混全帯幺九", "一気通貫", "三色同順", "三色同刻",
        "三槓子", "対々和", "三暗刻", "小三元", "混老頭", "二盃口", "純全帯幺九", "混一色", "清一色",
        "人和", "天和", "地和", "大三元", "四暗刻", "四暗刻単騎", "字一色", "緑一色", "清老頭",
        "九蓮宝燈", "純正九蓮宝燈", "国士無双", "国士無双13面待ち", "大四喜", "小四喜", "四槓子",
        "ドラ", "裏ドラ", "赤ドラ"]
LIMIT = ["", "満貫", "跳満", "倍満", "三倍満", "役満"]
DRAW = {"yao9": "九種九牌", "reach4": "四家立直", "ron3": "三家和了", "kaze4": "四風連打", "kan4": "四槓散了",
        "nm": "流し満貫"}


def meld_groups(codes):
    groups = []
    for m in codes:
        removed, shown, new_meld, _ = decode_meld(int(m))
        if new_meld:
            groups.append(list(shown))
        elif shown:                      # 加槓: ポンに1枚足す
            for g in groups:
                if len(g) == 3 and g[0] == shown[0] == g[1]:
                    g.append(shown[0])
                    break
    return groups


def make_result(r):
    """局の結果(和了/流局)を画面用にまとめる。結果が無ければ None。"""
    res = [a for a in r["results"] if a.get("attrs")]
    if not res:
        return None
    at0 = res[0]["attrs"]
    sc = [int(v) * 100 for v in at0["sc"].split(",")]
    before = sc[0::2]
    delta = [0, 0, 0, 0]
    for a in res:
        for i, d in enumerate(a["delta"]):
            delta[i] += d
    out = {"oya": r.get("oya", 0), "before": before, "delta": delta, "after": [b + d for b, d in zip(before, delta)],
           "honba": int(at0["ba"].split(",")[0]), "sticks": int(at0["ba"].split(",")[1]), "label": r["label"]}
    owari = next((a["attrs"]["owari"] for a in res if "owari" in a["attrs"]), None)
    if owari:
        o = owari.split(",")
        out["final"] = {"scores": [int(v) * 100 for v in o[0::2]], "pt": [float(v) for v in o[1::2]]}
    if res[0]["tag"] == "AGARI":
        out["kind"] = "agari"
        wins = []
        for a in res:
            at = a["attrs"]
            fu, pts, lim = [int(v) for v in at["ten"].split(",")]
            ys = [int(v) for v in at.get("yaku", "").split(",") if v != ""]
            yaku = [[YAKU[ys[i]], ys[i + 1]] for i in range(0, len(ys), 2) if ys[i + 1] > 0]
            ym = [YAKU[int(v)] for v in at.get("yakuman", "").split(",") if v != ""]
            hai = sorted(int(v) for v in at["hai"].split(","))
            machi = int(at["machi"])
            closed = list(hai)
            if machi in closed:
                closed.remove(machi)
            wins.append({
                "who": a["who"], "from": a["from"], "tsumo": a["who"] == a["from"], "fu": fu, "pts": pts,
                "limit": LIMIT[lim] if lim < len(LIMIT) else "", "yaku": yaku, "yakuman": ym,
                "han": sum(h for _, h in yaku),
                "hand": [i // 4 for i in closed], "hand_red": [i // 4 for i in closed if i in AKA_IDS],
                "win": machi // 4, "win_red": machi in AKA_IDS,
                "melds": meld_groups(at["m"].split(",")) if at.get("m") else [],
                "meldd": build_meld_list(at["m"].split(",")) if at.get("m") else [],
                "dora": [int(v) // 4 for v in at.get("doraHai", "").split(",") if v != ""],
                "ura": [int(v) // 4 for v in at.get("doraHaiUra", "").split(",") if v != ""],
            })
        out["wins"] = wins
    else:
        out["kind"] = "draw"
        at = res[0]["attrs"]
        out["draw"] = DRAW.get(at.get("type", ""), "荒牌平局")
        out["tenpai"] = [f"hai{i}" in at for i in range(4)]
    return out


def enc_river(river):
    """河の牌を整数1つにまとめる: 牌の種類 | 赤<<6 | リーチ宣言牌<<7 | 鳴かれた<<8 | ツモ切り<<9"""
    return [e[0] | (e[1] << 6) | (e[2] << 7) | (e[3] << 8) | (e[4] << 9) for e in river]


def analyse(ev, me, r, model, pfm, rrows, crows):
    """1つの打牌(席 me)について、AIの判断とコメントをまとめる。"""
    if hasattr(model, "infer"):          # numpy版(torch無し)
        probs, fold_p = model.infer(ev["x"], ev["f"])
    else:
        with torch.no_grad():
            scores, fold_logit = model(torch.from_numpy(ev["x"])[None], torch.from_numpy(ev["f"])[None])
        probs, fold_p = torch.softmax(scores, dim=1)[0].tolist(), torch.sigmoid(fold_logit).item()
    ctx, x, a = ev["ctx"], ev["x"], ev["actual"]
    draw = ev["draw"]
    c = x[0].astype(int).copy()
    if draw is not None and c[draw] > 0:
        c[draw] -= 1
    hand = [t for t in range(34) for _ in range(c[t])]
    cand = sorted(set(hand + ([draw] if draw is not None else [])))
    top = sorted(cand, key=lambda t: -probs[t])[:3]
    ai = top[0]
    sit_lines, info = situation.situation_report(ctx)
    comments = []
    lines = []
    if a == ai:
        lines.append(f"AIの1位と一致しました({probs[a]:.0%})。")
        lines += reasons.explain_choice(ev, a, review.style_potential)
    else:
        lines.append(f"AIは{NAMES[ai]}({probs[ai]:.0%})を推奨。あなたの{NAMES[a]}は{probs[a]:.0%}でした。")
        if x[0][a] > 0 and x[0][ai] > 0:
            lines += reasons.compare(ev, ai, a, review.style_potential)[0]
    comments.append({"h": "打牌", "t": lines})
    pf = review.push_fold(ev, probs, fold_p, pfm)
    if pf:
        comments.append({"h": f"押し引き({pf['verdict']})", "t": [review.explain({"actual": a, "pf": pf})]})
    if ev["f"][0] == 0 and any(ev["f"][k] for k in range(1, 4)):       # コンボ理論は、場面が限られるときだけ出す
        lv = [int(x[13][t]) - 1 for t in range(34) if x[0][t] > 0]
        mode = None
        if pf is None and lv and min(lv) > 2 and fold_p >= review.AI_FOLD:
            mode = "beta"                                            # ベタオリ推奨だが安全牌がない
        elif pf and review.AI_PUSH < pf["fold_p"] < review.AI_FOLD:
            mode = "mawashi"                                         # 回し打ち推奨
        if mode:
            cl = tactics.combo_report(ev, a, probs, mode)
            if cl:
                comments.append({"h": "コンボ理論(" + ("ベタオリ推奨・安全牌なし" if mode == "beta" else "回し打ち推奨") + ")", "t": cl})
    st = review.style_advice(ev, probs, info)
    if st:
        comments.append({"h": "手組み", "t": [
            f"オーラスで満貫級の打点が必要な局面。{NAMES[st['top']]}を切ると打点の見込みを残せます"
            f"({', '.join(st['diffs'])})。{shanten_text(st['sh'])}は同じです。"]})
    for rr in rrows:
        e2 = rr["ev"]
        if rr["round"] == r["label"] and e2["junme"] == ev["junme"] and e2["who"] == me:
            comments.append({"h": f"リーチ判断({rr['verdict']})",
                             "t": [f"あなたは{'リーチ' if rr['label'] else 'ダマ'}。"] + review_decisions.explain_riichi(rr)})
    for cr in crows:
        e2 = cr["ev"]
        if cr["round"] == r["label"] and e2["junme"] + 1 == ev["junme"]:
            if e2["label"] > 0 or cr["verdict"] != "問題なし" or cr["p_call_adj"] >= 0.3:
                mine = review_decisions.CALL_NAMES[e2["label"]]
                comments.append({"h": f"鳴き判断({cr['verdict']}) 自分: {mine}", "t": review_decisions.explain_call(cr)})
    rl = reads.read_lines(ev, me, r.setdefault("_shown", set()))
    if rl:
        comments.append({"h": "他家の河の読み", "t": rl})
    comments.append({"h": "点数状況", "t": sit_lines[:3]})
    flag = (a != ai and probs[a] < 0.15) or bool(pf and pf["verdict"] != "問題なし") or \
        any(c_["h"].startswith(("リーチ判断(ダマ推奨", "リーチ判断(リーチ推奨", "鳴き判断(鳴")) for c_ in comments)
    v = ev["viz"]
    hands = [expand(v["hands"][k]) for k in range(4)]
    me_hand_red = [] if v["draw_red"] else list(v["reds"][me])
    if draw is not None:
        hands[me].remove(draw)          # ツモ牌は別に出す
    return {
        "me": me, "t": v["t"], "junme": ev["junme"], "hand": hand, "draw": draw,
        "draw_red": bool(v["draw_red"]), "hand_red": me_hand_red, "actual": a,
        "prob": {str(t): round(probs[t], 3) for t in cand}, "top": top,
        "dora": expand(x[5]),
        "hands": hands, "melds": [expand(v["melds"][k]) for k in range(4)], "meldl": v["meld_list"],
        "reds": v["reds"], "rivers": [enc_river(rv) for rv in v["rivers"]],
        "riichi_seats": v["riichi_seats"],
        "riichi_now": bool(ev["riichi"]),
        "scores": ctx["scores"], "oya": ctx["oya"], "honba": ctx["honba"], "sticks": ctx["sticks"],
        "comments": comments, "flag": bool(flag), "match": a == ai,
    }


def build_game(model, models, path, target, progress=None):
    """progress(済んだ数, 全部の数) を渡すと、解析の途中経過を知らせる(ブラウザ版の進み具合表示用)。"""
    root = load(path)
    names = get_names(root)
    me0 = review.pick_seat(names, target)
    year = review.year_of(path)
    rounds = parse_game(root, year, trace=True)
    dec = parse_decisions(root, year, full=True)
    riichi_model, call_model, om, cm = models["rc"]
    pfm = models["pf"]
    out_rounds = [{"label": r["label"], "broken": r["broken"], "steps": [[], [], [], []], "result": make_result(r)}
                  for r in rounds]
    total = sum(len(r["events"]) for r in rounds if not r["broken"])
    done = 0
    for seat in range(4):
        rrows = review_decisions.judge_riichi(riichi_model, dec, seat, om) if riichi_model is not None else []
        crows = review_decisions.judge_call(call_model, dec, seat, cm) if call_model is not None else []
        for r, o in zip(rounds, out_rounds):
            if r["broken"]:
                continue
            for ev in r["events"]:
                if ev["who"] == seat:
                    o["steps"][seat].append(analyse(ev, seat, r, model, pfm, rrows, crows))
                    done += 1
                    if progress and done % 5 == 0:
                        progress(done, total)
    return {"title": os.path.basename(path), "names": list(names), "me": me0, "rounds": out_rounds}


def make_html(games):
    """games(build_game の結果のリスト)から、レビュー画面のHTML文字列を作る。"""
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, "viewer_template.html"), encoding="utf-8").read()
    for key, fn in (("__TILES__", "tiles.png"), ("__TABLE__", "table.png")):
        raw = open(os.path.join(here, "assets", fn), "rb").read()
        tpl = tpl.replace(key, "data:image/png;base64," + base64.b64encode(raw).decode())
    data = json.dumps({"games": games, "tiles": NAMES}, ensure_ascii=False, separators=(",", ":"))
    return tpl.replace("__DATA__", data.replace("</", "<\\/"))


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    model_path, out, target, logs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    if model_path.endswith(".npz"):           # numpy版(torch・scikit-learn不要)
        import npnet
        model, models = npnet.load_np(model_path)
    else:
        model = DiscardNet()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models = {"rc": review_decisions.load_models(model_path), "pf": review_decisions.load_pf_model(model_path)}
    games = []
    for p in logs:
        print("解析中:", p)
        games.append(build_game(model, models, p, target))
    open(out, "w", encoding="utf-8").write(make_html(games))
    n = sum(len(st) for g in games for r in g["rounds"] for st in r["steps"][g["me"]:g["me"] + 1])
    print(f"{out} を作りました({len(games)}件の牌譜、{n}打)。ブラウザで開いてください。")


if __name__ == "__main__":
    main()
