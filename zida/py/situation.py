"""点数状況(順位・点差・残り局数)から、「今どれだけ攻めるべきか」「何点の手が要るか」を求める。

考え方:
  ・最終的な順位ごとのpt(順位点)の期待値を「順位期待値」とする。
  ・オーラス(残り0局)は、局の終了時の点数でそのまま順位が決まる。
  ・それ以前は、残りの局数ぶんの点数のぶれ(1局あたり約5,500点の標準偏差)を考えて期待値を求める。
  ・和了(ツモ・ロン)と放銃で点数がどう動くかを、供託・本場も含めて計算し、順位期待値の変化を見る。

使えるもの:
  situation_report(ctx)       … 状況の説明文(行のリスト)と、数値を返す
  win_gain(ctx, ...)           … 和了したときの順位期待値の増え
  need_for_rank_up(ctx)        … (オーラス)順位が上がる最低打点
ctx は decisions.py の _Round.ctx() の dict。
"""
import math
from functools import lru_cache

import numpy as np

from scoring import level_name, payments

# 順位点(1着,2着,3着,4着)。鳳凰卓の段位ptに合わせてある。卓によって変えること。
#   特上卓: (60, 30, 0, -90)   上級卓: (30, 15, 0, -30) など
PLACE_PT = (90.0, 45.0, 0.0, -135.0)
SIGMA_ROUND = 5500.0          # 1局あたりの点数の変動の目安(標準偏差)
WIND = "東南西"

_rng = np.random.default_rng(12345)
_Z = _rng.standard_normal((6000, 4))


def round_label(ctx):
    i = ctx["round_idx"]
    return f"{WIND[min(i // 4, 2)]}{i % 4 + 1}局"


def rounds_after(ctx):
    """この局が終わった後に残る局数。0ならこの局が最終局(オーラス)。"""
    return max(ctx["last_round"] - ctx["round_idx"], 0)


def rank_of(scores, me):
    """順位(1〜4)。同点は起家(席番号が小さい方)が上。"""
    return 1 + sum(1 for q in range(4) if scores[q] > scores[me] or (scores[q] == scores[me] and q < me))


def expected_pt(scores, me, rounds_left, pt=None):
    """残り rounds_left 局の後の、順位pt期待値。0ならそのまま順位で決まる。"""
    pt = pt or PLACE_PT
    if rounds_left <= 0:
        return pt[rank_of(scores, me) - 1]
    s = np.array(scores, dtype=float) + SIGMA_ROUND * math.sqrt(rounds_left) * _Z
    r = (s > s[:, [me]]).sum(1)
    return float(np.array(pt)[r].mean())


# ---------------------------------------------------------------
# 和了・放銃の点数の動き
# ---------------------------------------------------------------
def ladder(dealer):
    """よくある打点の一覧(小さい順)。要素: (表示, 翻, 符, ロンの点数, ツモの合計点)"""
    seen, out = set(), []
    cand = [(h, f) for h in (1, 2, 3, 4) for f in (30, 40, 50, 60)] + [(5, 30), (6, 30), (8, 30), (11, 30), (13, 30)]
    for han, fu in cand:
        pay = payments(han, fu, dealer)
        if pay["ron"] in seen:
            continue
        seen.add(pay["ron"])
        lv = level_name(han, fu)
        out.append((lv or f"{han}翻{fu}符", han, fu, pay["ron"], pay["tsumo_total"]))
    out.sort(key=lambda r: r[3])
    return out


def _after_win(ctx, han, fu, tsumo, payer=None, riichi=False):
    """自分が(han,fu)で和了したあとの、4人の点数。riichi=Trueなら、自分のリーチ棒(1000点)を先に出した状態。"""
    me, oya = ctx["me"], ctx["oya"]
    sc = list(ctx["scores"])
    sticks = ctx["sticks"] + (1 if riichi else 0)
    if riichi:
        sc[me] -= 1000
    honba = ctx["honba"]
    dealer = ctx["seat"] == 0
    pay = payments(han, fu, dealer)
    if tsumo:
        total = 0
        for q in range(4):
            if q == me:
                continue
            amt = pay["tsumo_each"][0] if (dealer or q == oya) else pay["tsumo_each"][1]
            amt += 100 * honba
            sc[q] -= amt
            total += amt
        sc[me] += total + 1000 * sticks
    else:
        amt = pay["ron"] + 300 * honba
        sc[payer] -= amt
        sc[me] += amt + 1000 * sticks
    return sc


def _after_win_points(ctx, pts, tsumo, payer=None, riichi=False):
    """点数(ロンなら合計点、ツモなら合計点)だけが分かっているときの和了後の点数。ツモは親子比を2:1:1(親ならオール)で配分。"""
    me, oya = ctx["me"], ctx["oya"]
    sc = list(ctx["scores"])
    sticks = ctx["sticks"] + (1 if riichi else 0)
    if riichi:
        sc[me] -= 1000
    honba = ctx["honba"]
    if tsumo:
        dealer = ctx["seat"] == 0
        unit = pts / (3.0 if dealer else 4.0)
        total = 0
        for q in range(4):
            if q == me:
                continue
            amt = unit * (1 if dealer else (2 if q == oya else 1)) + 100 * honba
            amt = -(-amt // 100) * 100
            sc[q] -= amt
            total += amt
        sc[me] += total + 1000 * sticks
    else:
        amt = pts + 300 * honba
        sc[payer] -= amt
        sc[me] += amt + 1000 * sticks
    return sc


def _after_loss(ctx, pts, to):
    me = ctx["me"]
    sc = list(ctx["scores"])
    amt = pts + 300 * ctx["honba"]
    sc[me] -= amt
    sc[to] += amt + 1000 * ctx["sticks"]
    return sc


def win_gain(ctx, pts, tsumo, riichi=False):
    """pts点の和了をしたときの、順位期待値の増え(相手は全員の平均)。ロンは3人のうち誰から出るかで平均する。"""
    me, R = ctx["me"], rounds_after(ctx)
    e0 = expected_pt(ctx["scores"], me, R)
    if tsumo:
        return expected_pt(_after_win_points(ctx, pts, True, riichi=riichi), me, R) - e0
    gains = [expected_pt(_after_win_points(ctx, pts, False, q, riichi), me, R) - e0 for q in range(4) if q != me]
    return float(np.mean(gains))


def loss_cost(ctx, pts):
    """pts点を放銃したときの、順位期待値の減り(相手は全員の平均)。正の数で返す。"""
    me, R = ctx["me"], rounds_after(ctx)
    e0 = expected_pt(ctx["scores"], me, R)
    return float(np.mean([e0 - expected_pt(_after_loss(ctx, pts, q), me, R) for q in range(4) if q != me]))


# ---------------------------------------------------------------
# オーラスの「必要打点」
# ---------------------------------------------------------------
def need_for_rank_up(ctx):
    """(最終局)順位が1つ以上上がる最低打点。戻り値 dict:
        tsumo : 表示 or None(ツモで足りる最低の打点。届かないなら None)
        ron   : {相手の席: (表示, 点数) or None}
    今がトップなら None を返す。"""
    me = ctx["me"]
    r0 = rank_of(ctx["scores"], me)
    if r0 == 1:
        return None
    lad = ladder(ctx["seat"] == 0)
    out = {"tsumo": None, "ron": {}}
    for label, han, fu, ron, tsu in lad:
        if rank_of(_after_win(ctx, han, fu, True), me) < r0:
            out["tsumo"] = (label, tsu)
            break
    for q in range(4):
        if q == me:
            continue
        out["ron"][q] = None
        for label, han, fu, ron, tsu in lad:
            if rank_of(_after_win(ctx, han, fu, False, q), me) < r0:
                out["ron"][q] = (label, ron)
                break
    return out


def min_need_points(ctx):
    """順位が上がる最低の打点(ロンの点数)。誰から出てもいいとき最も条件が緩い場合と、ツモの場合。無ければ None。"""
    nd = need_for_rank_up(ctx)
    if not nd:
        return None
    rons = [v[1] for v in nd["ron"].values() if v]
    return {"ron_min": min(rons) if rons else None, "tsumo": nd["tsumo"][1] if nd["tsumo"] else None,
            "detail": nd}


# ---------------------------------------------------------------
# 攻め守りの度合い
# ---------------------------------------------------------------
def _aggression(ctx):
    """攻め(+1)〜守り(-1)の度合いと、その根拠の数値。
        up   … 満貫を和了したときの順位期待値の増え(ツモ/ロン平均)
        down … 5,200点を放銃したときの順位期待値の減り"""
    dealer = ctx["seat"] == 0
    mg = 12000 if dealer else 8000          # 満貫の合計点
    up = 0.5 * win_gain(ctx, mg, False) + 0.5 * win_gain(ctx, mg, True)
    down = loss_cost(ctx, 7700 if dealer else 5200)
    tot = up + down
    if tot < 1e-9:
        return {"aggr": 0.0, "up": up, "down": down}
    raw = (up - down) / tot
    weight = min(1.0, tot / 20.0)       # 順位への影響が小さい局面(序盤など)では、偏らせない
    return {"aggr": float(raw * weight), "up": up, "down": down}


def stance_text(a):
    x = a["aggr"]
    if x >= 0.5:
        return "攻め重視(上を狙える一方、放銃で失うものが小さい)"
    if x >= 0.2:
        return "やや攻め寄り"
    if x <= -0.5:
        return "守り重視(順位を守りたい。放銃の痛手が大きい)"
    if x <= -0.2:
        return "やや守り寄り"
    return "どちらにも偏らない"


def _situation_report(ctx):
    """状況の説明(行のリスト)と数値を返す。"""
    me, sc = ctx["me"], ctx["scores"]
    R = rounds_after(ctx)
    rank = rank_of(sc, me)
    order = sorted(range(4), key=lambda q: (-sc[q], q))
    above = [q for q in order if rank_of(sc, q) < rank]
    below = [q for q in order if rank_of(sc, q) > rank]
    lines = []
    head = f"{round_label(ctx)}{ctx['honba']}本場" + ("(オーラス)" if R == 0 else "") + \
        f"  {rank}着 {sc[me]:,}点"
    if rank > 1:
        head += f" / トップと{sc[order[0]] - sc[me]:,}点差"
    if above:
        head += f" / 直上と{sc[above[-1]] - sc[me]:,}点差"
    if below:
        head += f" / 直下に{sc[me] - sc[below[0]]:,}点リード"
    lines.append(head)
    info = {"rank": rank, "R": R}
    if R == 0:
        nd = need_for_rank_up(ctx)
        if nd:
            parts = []
            if nd["tsumo"]:
                parts.append(f"ツモなら{nd['tsumo'][0]}({nd['tsumo'][1]:,}点)以上")
            else:
                parts.append("ツモでも届かない")
            rons = {q: v for q, v in nd["ron"].items() if v}
            if rons:
                lo = min(v[1] for v in rons.values())
                hi = max(v[1] for v in rons.values())
                if lo == hi:
                    parts.append(f"ロンなら{lo:,}点以上")
                else:
                    parts.append(f"ロンなら相手次第で{lo:,}〜{hi:,}点以上")
            else:
                parts.append("ロンでは届かない")
            lines.append(f"順位を上げるのに必要な打点: " + " / ".join(parts))
        else:
            gap = sc[me] - sc[order[1]]
            lines.append(f"トップ目。2着と{gap:,}点差 → 和了より放銃回避を優先して逃げ切る局面")
        # ラス回避の余裕
        if rank == 3 and below:
            lines.append(f"ラス回避: 4着と{sc[me] - sc[below[0]]:,}点差。放銃すると4着に落ちるか注意")
        if rank == 4:
            lines.append("4着。3着になるための打点が必要" if nd else "")
    a = aggression(ctx)
    info.update(a)
    info["need"] = min_need_points(ctx) if R == 0 else None
    lines.append(f"状況の方針: {stance_text(a)}")
    info["policy"] = None
    if R == 0:
        need = info["need"]
        if rank == 1:
            info["policy"] = "defend"
            lines.append("手組みの方針: 逃げ切り。和了を急がず、放銃回避を最優先")
        elif need:
            target = need["ron_min"] or need["tsumo"]
            if target is None:
                info["policy"] = "balance"
            elif target >= 7700:
                info["policy"] = "value"
                lines.append(f"手組みの方針: {target:,}点以上が必要 → 満貫級の打点を優先"
                             "(向聴数が同じなら、ドラ・役牌・染め手・リーチを活かせる形を残す)")
            elif target <= 3900:
                info["policy"] = "speed"
                lines.append(f"手組みの方針: {target:,}点以上で順位が上がる → 安い手でもよいので速度優先"
                             "(鳴き・ダマ・安手の和了も有効)")
            else:
                info["policy"] = "balance"
    return [l for l in lines if l], info


# ---- 同じ局面を何度も計算しないための記憶 ----
_KEYS = ("round_idx", "last_round", "honba", "sticks", "me", "oya", "seat")


def _key(ctx):
    return tuple(ctx[k] for k in _KEYS) + (tuple(ctx["scores"]),)


def _from_key(key):
    ctx = dict(zip(_KEYS, key[:-1]))
    ctx["scores"] = list(key[-1])
    return ctx


@lru_cache(maxsize=4096)
def _aggression_cached(key):
    return _aggression(_from_key(key))


@lru_cache(maxsize=4096)
def _report_cached(key):
    lines, info = _situation_report(_from_key(key))
    return tuple(lines), info


def aggression(ctx):
    return dict(_aggression_cached(_key(ctx)))


def situation_report(ctx):
    lines, info = _report_cached(_key(ctx))
    return list(lines), dict(info)
