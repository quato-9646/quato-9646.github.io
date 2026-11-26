"""他家の河(捨て牌)から、染め手の気配と、切る順序の不自然さを読み取って言葉にする。
場がある程度進んでから(自分の8巡目以降)だけ使う。あくまで推測で、断定はしない。"""
from rules import SEAT_NAMES

SUITS = ["萬子", "筒子", "索子"]
MIN_JUNME = 8


def _suit_counts(river):
    cnt, honor = [0, 0, 0], 0
    for e in river:
        t = e[0]
        if t >= 27:
            honor += 1
        else:
            cnt[t // 9] += 1
    return cnt, honor


def _meld_info(counts):
    """副露の牌(種類ごとの枚数の表)から、数牌のある色の集合と、字牌があるかを返す。"""
    suits = {t // 9 for t in range(27) if counts[t] > 0}
    honor = any(counts[t] > 0 for t in range(27, 34))
    return suits, honor


def some_read(viz, k):
    """席 k の染め手の見込み。なければ None。(色の番号, 強さの文)"""
    rv = viz["rivers"][k]
    if len(rv) < 8:
        return None
    cnt, honor = _suit_counts(rv)
    msuits, mhonor = _meld_info(viz["melds"][k])
    order = sorted(range(3), key=lambda s: cnt[s])
    s, s2 = order[0], order[1]
    if cnt[s] > 1 or cnt[s2] < 3:
        return None
    others = sum(cnt) - cnt[s]
    if others < 6 or cnt[s] > 0.15 * len(rv):
        return None
    if msuits - {s}:                  # 別の色の副露があれば、染めていない
        return None
    nm = SUITS[s]
    strong = bool(msuits) and msuits == {s}
    if honor >= 3 and cnt[s] == 0:
        kind = "清一色"
        txt = f"{nm}を1枚も切らず、字牌も{honor}枚切っていて、清一色(または{nm}の染め)の気配があります。"
    else:
        kind = "混一色"
        txt = f"{nm}をほとんど切らず、ほかの色を{others}枚切っていて、混一色({nm})の気配があります。"
        if honor <= 1:
            txt += "字牌をあまり切らないので、字牌も使っている可能性があります。"
    if strong:
        txt += f"副露も{nm}だけなので、かなり有力です。"
    return s, kind, txt


def order_reads(viz, k):
    """手出しで中張牌(4〜6)を切った直後に、手出しで端牌・字牌を切っている箇所。(位置, 文)のリスト。"""
    rv = viz["rivers"][k]
    out = []
    ted = [(i, e[0]) for i, e in enumerate(rv) if not e[4]]
    for a in range(len(ted)):
        i, t = ted[a]
        if t >= 27 or not (3 <= t % 9 <= 5):
            continue
        for b in range(a + 1, min(a + 2, len(ted))):      # 直後の手出しだけ見る
            j, u = ted[b]
            if u >= 27 or u % 9 in (0, 8):
                out.append((j, i, t, u))
                break
    return out


def read_lines(ev, me, shown):
    """このとき表示する、他家の河の読み(文のリスト)。shownは同じ内容を繰り返さないための記録。"""
    if ev["junme"] < MIN_JUNME:
        return []
    from mjcommon import NAMES
    viz = ev["viz"]
    lines = []
    for k in range(4):
        if k == me:
            continue
        who = SEAT_NAMES[(k - me) % 4]
        r = some_read(viz, k)
        if r:
            s, kind, txt = r
            key = ("some", k, s, kind)
            if key not in shown:
                shown.add(key)
                lines.append(f"{who}: {txt}")
        for j, i, t, u in order_reads(viz, k):
            if j < len(viz["rivers"][k]) - 4:     # 古いものは触れない(最近の4打牌以内だけ)
                continue
            key = ("order", k, j)
            if key in shown:
                continue
            shown.add(key)
            lines.append(f"{who}: 手出しで{NAMES[t]}(中張牌)を切った後に、手出しで{NAMES[u]}"
                         f"({'字牌' if u >= 27 else '端牌'})を切っています。真ん中の牌から先に払って端を残すのは、"
                         "普通と逆の順序です。端の牌が必要な形(チャンタ系・役牌・七対子など)か、"
                         "すでに形が決まっていて中張牌が余った可能性があります。")
    return lines
