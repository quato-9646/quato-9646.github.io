"""通常の打牌について、AIの推奨牌と自分の牌を比べて、違いを数値から言葉にする。
学習モデルの確率ではなく、向聴数・受け入れ枚数・打点の見込み・安全度を計算して理由にする。"""
import math

from mjcommon import NAMES
from rules import LEVEL_NAMES, shanten, shanten_text
import tactics

DORA_YAKU = (27, 28, 29, 30, 31, 32, 33)


def _visible(x):
    v = x[0] + x[1] + x[2] + x[3] + x[4] + x[6] + x[7] + x[8] + x[9]
    return [int(a) for a in v]


def ukeire(ev, t):
    """tを切った後の手が、1枚引くと向聴数が下がる牌の (種類, 残り枚数合計, 牌リスト)。"""
    x = ev["x"]
    hand = [int(v) for v in x[0]]
    nm = int(round(float(x[6].sum()) / 3))
    hand[t] -= 1
    base = shanten(hand, nm)
    vis = _visible(x)
    tiles, total = [], 0
    for u in range(34):
        if hand[u] >= 4:
            continue
        hand[u] += 1
        if shanten(hand, nm) < base:
            left = max(0, 4 - vis[u])
            if left > 0:
                tiles.append(u)
                total += left
        hand[u] -= 1
    return len(tiles), total, tiles


def role(ev, t):
    """tが手の中で何の役割か(短い言葉)。"""
    x = ev["x"]
    hand = [int(v) for v in x[0]]
    c = hand[t]
    isdora = x[5][t] > 0
    if c >= 3:
        return "刻子の一部"
    if c == 2:
        return "対子"
    if t >= 27:
        return "孤立した役牌" if t >= 31 else "孤立した字牌"
    n = t % 9
    base = t - n
    near = {d: hand[base + n + d] for d in (-2, -1, 1, 2) if 0 <= n + d < 9 and hand[base + n + d] > 0}
    if not near:
        return "孤立した" + ("端牌" if n in (0, 8) else "中張牌")
    has = lambda d: d in near
    if (has(-1) and has(1)) or (has(-1) and has(-2)) or (has(1) and has(2)):
        return "順子の一部"
    if has(-1) or has(1):
        return "辺張の搭子" if (n in (1, 7) and has(-1 if n == 7 else 1)) or (n in (0, 8)) else "両面の搭子"
    return "嵌張の搭子"


def tile_desc(ev, t):
    d = NAMES[t] + ("(ドラ)" if ev["x"][5][t] > 0 else "")
    return f"{d}は{role(ev, t)}"


def compare(ev, rec, act, style_potential, riichi_level=None):
    """rec(AI推奨)とact(自分)を比べた理由の文のリストと、決め手の分類を返す。"""
    x = ev["x"]
    sh = lambda t: int(x[14][t]) - 1
    out, kind = [], "好み"
    sr, sa = sh(rec), sh(act)
    ur, ua = ukeire(ev, rec), ukeire(ev, act)
    pr, pa = style_potential(ev, rec), style_potential(ev, act)
    out.append(f"{tile_desc(ev, rec)}、{tile_desc(ev, act)}。")
    if sr < sa:
        kind = "向聴数"
        out.append(f"{NAMES[act]}を切ると{shanten_text(sa)}、{NAMES[rec]}なら{shanten_text(sr)}で、手が1つ進みます。")
    elif sr > sa:
        kind = "向聴数(自分が進む)"
        out.append(f"{NAMES[act]}の方が{shanten_text(sa)}で手が進みます(AIの推奨は{shanten_text(sr)})。形より安全や打点を見た選択かもしれません。")
    else:
        diff = ur[1] - ua[1]
        if diff >= 3:
            kind = "受け入れ"
            out.append(f"同じ{shanten_text(sr)}でも、{NAMES[rec]}切りは受け入れ{ur[1]}枚({ur[0]}種)、"
                       f"{NAMES[act]}切りは{ua[1]}枚({ua[0]}種)で、{diff}枚多く、和了に近づきやすい形です。")
        elif diff <= -3:
            kind = "受け入れ(自分が広い)"
            out.append(f"受け入れは{NAMES[act]}切りの方が{-diff}枚多い({ua[1]}枚 対 {ur[1]}枚)ので、速度だけ見ればあなたの打牌も悪くありません。")
        else:
            out.append(f"同じ{shanten_text(sr)}で受け入れも近い({NAMES[rec]}切り{ur[1]}枚 / {NAMES[act]}切り{ua[1]}枚)です。")
        dv = pr[0] - pa[0]
        if abs(dv) >= 0.7:
            names = [f"{k} {pr[1][k] - pa[1][k]:+.1f}" for k in pr[1] if abs(pr[1][k] - pa[1][k]) >= 0.3]
            if dv > 0:
                if kind == "好み":
                    kind = "打点"
                out.append(f"打点の見込みは{NAMES[rec]}切りの方が高く残ります({', '.join(names)})。")
            else:
                out.append(f"打点の見込みは{NAMES[act]}切りの方が高く残ります({', '.join(names)})。")
    if kind == "好み":                         # 数値で差が出ない場面は、定石で説明する
        rr, ra = role(ev, rec), role(ev, act)
        iso = lambda r: r.startswith("孤立")
        if iso(rr) and not iso(ra):
            kind = "定石(孤立牌から)"
            out.append(f"数値の差は小さいですが、孤立した牌から切り、{ra.replace('の一部','')}を残して形の柔軟さを保つのが定石です。")
        elif iso(rr) and iso(ra):
            kind = "定石(孤立牌の優先順)"
            edge = lambda t: t >= 27 or t % 9 in (0, 8)
            if edge(rec) and not edge(act):
                out.append(f"どちらも孤立牌ですが、端牌・字牌は面子になりにくく、中張牌は両面に育ちやすいので、{NAMES[rec]}を先に切るのが定石です。")
            elif edge(act) and not edge(rec):
                out.append(f"どちらも孤立牌で、定石では{NAMES[act]}のような端牌・字牌を先に切る場面です。AIが{NAMES[rec]}を勧めるのは、数値上の差が小さく、プロの傾向に近い方を挙げているためで、あなたの打牌も十分ありえます。")
            else:
                out.append(f"どちらも孤立牌です。AIは{NAMES[rec]}を先に切る傾向ですが、数値上の差は小さい場面です。")
        else:
            out.append("数値(向聴数・受け入れ・打点)では差が出ない場面で、AIは牌譜上のプロの傾向に従っています。")
    if x[13][act] > 0 or x[13][rec] > 0:      # 誰かがリーチ中
        la, lr = int(x[13][act]) - 1, int(x[13][rec]) - 1
        if la > lr:
            if kind == "好み":
                kind = "安全度"
            out.append(f"リーチ中なので安全度も差が出ます。{NAMES[rec]}は{LEVEL_NAMES[lr]}、{NAMES[act]}は{LEVEL_NAMES[la]}です。")
        elif lr > la:
            out.append(f"安全度では{NAMES[act]}({LEVEL_NAMES[la]})の方が{NAMES[rec]}({LEVEL_NAMES[lr]})より安全です。")
        for tt in (rec, act):
            c = tactics.chance(ev, tt)
            if c and x[13][tt] - 1 >= 3:       # 無筋・半スジなど、危険な牌にだけ補足する
                out.append(c)
        ca, cr = tactics.combo_text(ev, act), tactics.combo_text(ev, rec)
        pa = max(tactics.combo_points(ev, act, k)[0] for k in range(1, 4) if ev["f"][k]) if ca else 0
        pr_ = max(tactics.combo_points(ev, rec, k)[0] for k in range(1, 4) if ev["f"][k]) if cr else 0
        if ca and cr:
            out.append(cr + " " + ca.replace("コンボ理論では、", ""))
            if pa > pr_:
                out.append(f"当たる組み合わせは{NAMES[act]}の方が{pa - pr_}ポイント多く、{NAMES[rec]}より放銃しやすい切り方です。")
            elif pr_ > pa:
                out.append(f"当たる組み合わせは{NAMES[rec]}の方が{pr_ - pa}ポイント多く、安全面だけならあなたの{NAMES[act]}の方が通りやすい牌です。")
    return out, kind


def explain_choice(ev, t, style_potential):
    """AIの推奨牌(または一致した牌)を切る理由を、数値で短く説明する。"""
    x = ev["x"]
    sh = int(x[14][t]) - 1
    n, total, _ = ukeire(ev, t)
    out = [f"{tile_desc(ev, t)}。{NAMES[t]}を切ると{shanten_text(sh)}、受け入れ{total}枚({n}種)。"]
    pot = style_potential(ev, t)
    keep = [f"{k}{v:+.1f}".replace("+", "") for k, v in pot[1].items() if v >= 0.7]
    if keep:
        out.append("残る打点の見込み: " + "・".join(keep) + "。")
    if x[13][t] > 0:
        out.append(f"リーチ中の相手に対して{NAMES[t]}は{LEVEL_NAMES[int(x[13][t]) - 1]}。")
        c = tactics.chance(ev, t)
        if c and x[13][t] - 1 >= 3:
            out.append(c)
        ct = tactics.combo_text(ev, t)
        if ct:
            out.append(ct)
            rk = tactics.combo_rank(ev, t)
            if rk and rk[1] >= 2:
                out.append(f"手の数牌{rk[1]}種のうち、危険な順で{rk[0]}番目です(手牌の中の最大{rk[4]}ポイント・最小{rk[3]}ポイント)。")
    return out
