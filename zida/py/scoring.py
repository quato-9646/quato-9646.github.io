"""打点の計算(役・符・翻・点数)。

手牌(34種の枚数)・副露・和了牌・場の条件から、その和了の点数を求める。
  evaluate(closed, melds, win, tsumo, ...) … 和了形の点数(役が無ければ None)
  tenpai_values(closed13, melds, waits, live, ...) … 聴牌の手の、待ち牌ごとの点数の平均など

牌の番号: 0-8 萬子 / 9-17 筒子 / 18-26 索子 / 27-30 東南西北 / 31-33 白發中。
副露(melds)は [(種類, 牌, 明か), ...]
    種類 'chi' : 牌は順子の一番小さい牌   'pon'   : 刻子の牌   'kan' : 槓子の牌
    明か : 明(チー・ポン・大明槓・加槓)=True / 暗槓=False
ルール: 赤ドラは枚数を dora に足して渡す。裏ドラ・一発は別に ura で渡す。
役満は13翻として扱う(ダブル役満は考えない)。
"""
from functools import lru_cache

YAOCHU = frozenset([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
TERMINAL = frozenset([0, 8, 9, 17, 18, 26])
GREEN = frozenset([19, 20, 21, 23, 25, 32])      # 2s 3s 4s 6s 8s 發
LEVELS = ["", "満貫", "跳満", "倍満", "三倍満", "役満"]


def _ceil100(v):
    return -(-v // 100) * 100


def base_points(han, fu):
    """基本点(符×2^(翻+2)。満貫以上は固定)。"""
    if han >= 13:
        return 8000
    if han >= 11:
        return 6000
    if han >= 8:
        return 4000
    if han >= 6:
        return 3000
    return min(fu * (1 << (han + 2)), 2000)


def payments(han, fu, dealer):
    """点数の支払い。戻り値 dict: ron(ロンの点数) / tsumo_each(親ならオール、子なら(親,子)) / tsumo_total"""
    b = base_points(han, fu)
    if dealer:
        each = _ceil100(b * 2)
        return {"ron": _ceil100(b * 6), "tsumo_each": (each, each), "tsumo_total": each * 3}
    pay_oya, pay_ko = _ceil100(b * 2), _ceil100(b)
    return {"ron": _ceil100(b * 4), "tsumo_each": (pay_oya, pay_ko), "tsumo_total": pay_oya + pay_ko * 2}


def level_name(han, fu):
    b = base_points(han, fu)
    if han >= 13:
        return "役満"
    if b >= 6000:
        return "三倍満"
    if b >= 4000:
        return "倍満"
    if b >= 3000:
        return "跳満"
    if b >= 2000:
        return "満貫"
    return ""


# ---------------------------------------------------------------
# 手牌の分解
# ---------------------------------------------------------------
def _decomps(counts):
    """門前部分(counts)を「面子+雀頭」に分ける全ての方法。要素: (面子のリスト, 雀頭)。
    面子は ('seq', 最小の牌) か ('trip', 牌)。"""
    c = list(counts)
    out = []

    def rec(i, groups, pair):
        while i < 34 and c[i] == 0:
            i += 1
        if i == 34:
            if pair is not None:
                out.append((tuple(groups), pair))
            return
        if c[i] >= 3:
            c[i] -= 3
            groups.append(("trip", i))
            rec(i, groups, pair)
            groups.pop()
            c[i] += 3
        if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            groups.append(("seq", i))
            rec(i, groups, pair)
            groups.pop()
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        if pair is None and c[i] >= 2:
            c[i] -= 2
            rec(i, groups, i)
            c[i] += 2

    rec(0, [], None)
    return out


def _wait_options(groups, pair, win):
    """和了牌がどの面子/雀頭を完成させたか、の解釈の一覧。(待ちの種類, 完成した面子の番号 or None)"""
    opts = []
    seen = set()
    if pair == win:
        opts.append(("tanki", None))
    for gi, (kind, t) in enumerate(groups):
        if (kind, t) in seen:
            continue
        if kind == "trip" and t == win:
            opts.append(("shanpon", gi)); seen.add((kind, t))
        elif kind == "seq" and t <= win <= t + 2:
            if win == t + 1:
                w = "kanchan"
            elif win == t:
                w = "penchan" if t % 9 == 6 else "ryanmen"
            else:
                w = "penchan" if t % 9 == 0 else "ryanmen"
            opts.append((w, gi)); seen.add((kind, t))
    return opts


# ---------------------------------------------------------------
# 1つの解釈の点数
# ---------------------------------------------------------------
def _score_decomp(groups, pair, wait, wgi, melds, win, tsumo, round_wind, seat_wind,
                  riichi, kuitan):
    # 面子の一覧: (種類, 牌, 明か, 槓か, 暗刻か)
    gs = []
    for gi, (kind, t) in enumerate(groups):
        completed_by_ron = (gi == wgi and kind == "trip" and not tsumo)
        gs.append((kind, t, completed_by_ron, False, kind == "trip" and not completed_by_ron))
    for kind, t, is_open in melds:
        if kind == "chi":
            gs.append(("seq", t, True, False, False))
        else:
            gs.append(("trip", t, is_open, kind == "kan", not is_open))
    menzen = not any(m[2] for m in melds)
    tiles_all = set()
    for kind, t, *_ in gs:
        tiles_all.update([t, t + 1, t + 2] if kind == "seq" else [t])
    tiles_all.add(pair)
    seqs = [g for g in gs if g[0] == "seq"]
    trips = [g for g in gs if g[0] == "trip"]
    yaku = []          # (名前, 翻)
    yakuman = []

    dragons = [t for t in (31, 32, 33) if any(g[1] == t for g in trips)]
    winds_t = [t for t in (27, 28, 29, 30) if any(g[1] == t for g in trips)]
    # ---- 役満 ----
    if len(dragons) == 3:
        yakuman.append("大三元")
    if len(winds_t) == 4:
        yakuman.append("大四喜")
    elif len(winds_t) == 3 and pair in (27, 28, 29, 30):
        yakuman.append("小四喜")
    if all(t >= 27 for t in tiles_all):
        yakuman.append("字一色")
    if all(t in TERMINAL for t in tiles_all):
        yakuman.append("清老頭")
    if all(t in GREEN for t in tiles_all):
        yakuman.append("緑一色")
    if sum(1 for g in trips if g[4]) == 4:
        yakuman.append("四暗刻")
    if sum(1 for g in gs if g[3]) == 4:
        yakuman.append("四槓子")
    if yakuman:
        return {"han": 13 * len(yakuman), "fu": 30, "yaku": [(n, 13) for n in yakuman], "yakuman": True}

    # ---- 通常の役 ----
    if riichi and menzen:
        yaku.append(("立直", 1))
    if tsumo and menzen:
        yaku.append(("門前清自摸和", 1))
    all_seq = not trips
    pair_yakuhai = pair in (31, 32, 33) or pair == 27 + round_wind or pair == 27 + seat_wind
    if menzen and all_seq and not pair_yakuhai and wait == "ryanmen":
        yaku.append(("平和", 1))
    if all(t not in YAOCHU for t in tiles_all) and (menzen or kuitan):
        yaku.append(("断幺九", 1))
    if menzen:
        seq_keys = sorted(g[1] for g in seqs)
        dup = sum(1 for t in set(seq_keys) if seq_keys.count(t) >= 2)
        if sum(1 for t in set(seq_keys) if seq_keys.count(t) >= 4) or dup == 2:
            yaku.append(("二盃口", 3))
        elif dup == 1:
            yaku.append(("一盃口", 1))
    for g in trips:
        t = g[1]
        n = 0
        if t in (31, 32, 33):
            n += 1
        if t == 27 + round_wind:
            n += 1
        if t == 27 + seat_wind:
            n += 1
        if n:
            yaku.append((f"役牌({'白發中'[t - 31] if t >= 31 else '東南西北'[t - 27]})", n))
    sq = {g[1] for g in seqs}
    for n in range(7):
        if all(s * 9 + n in sq for s in range(3)):
            yaku.append(("三色同順", 2 if menzen else 1))
            break
    for s in range(3):
        if all(s * 9 + k in sq for k in (0, 3, 6)):
            yaku.append(("一気通貫", 2 if menzen else 1))
            break
    all_trips = not seqs
    honors = any(t >= 27 for t in tiles_all)
    group_has_yaochu = all(
        (g[0] == "seq" and (g[1] % 9 == 0 or g[1] % 9 == 6)) or (g[0] == "trip" and g[1] in YAOCHU) for g in gs
    ) and pair in YAOCHU
    if all(t in YAOCHU for t in tiles_all):
        yaku.append(("混老頭", 2))
    elif group_has_yaochu:
        yaku.append(("混全帯幺九" if honors else "純全帯幺九", (2 if menzen else 1) + (0 if honors else 1)))
    if all_trips:
        yaku.append(("対々和", 2))
    if sum(1 for g in trips if g[4]) == 3:
        yaku.append(("三暗刻", 2))
    if sum(1 for g in gs if g[3]) == 3:
        yaku.append(("三槓子", 2))
    for n in range(9):
        if all(s * 9 + n in {g[1] for g in trips} for s in range(3)):
            yaku.append(("三色同刻", 2))
            break
    if len(dragons) == 2 and pair in (31, 32, 33):
        yaku.append(("小三元", 2))
    suits = {t // 9 for t in tiles_all if t < 27}
    if len(suits) == 1:
        if honors:
            yaku.append(("混一色", 3 if menzen else 2))
        else:
            yaku.append(("清一色", 6 if menzen else 5))
    if not yaku:
        return None

    # ---- 符 ----
    if menzen and all_seq and not pair_yakuhai and wait == "ryanmen":
        fu = 20 if tsumo else 30          # 平和
    else:
        fu = 20
        if menzen and not tsumo:
            fu += 10
        if tsumo:
            fu += 2
        for kind, t, op, is_kan, conc in gs:
            if kind != "trip":
                continue
            v = 4 if t in YAOCHU else 2
            if is_kan:
                v *= 4
            if conc:
                v *= 2
            fu += v
        if pair in (31, 32, 33):
            fu += 2
        if pair == 27 + round_wind:
            fu += 2
        if pair == 27 + seat_wind:
            fu += 2
        if wait in ("kanchan", "penchan", "tanki"):
            fu += 2
        if fu == 20 and not menzen:
            fu = 30
        fu = -(-fu // 10) * 10
    return {"han": sum(h for _, h in yaku), "fu": fu, "yaku": yaku, "yakuman": False}


def _special(closed, melds, win, tsumo, riichi, round_wind, seat_wind):
    """七対子・国士無双・九蓮宝燈(分解できない形/特殊な形)。"""
    res = []
    if not melds:
        pairs = [t for t in range(34) if closed[t] == 2]
        if len(pairs) == 7 and sum(closed) == 14:
            yaku = [("七対子", 2)]
            if riichi:
                yaku.append(("立直", 1))
            if tsumo:
                yaku.append(("門前清自摸和", 1))
            if all(t >= 27 for t in pairs):
                yaku = [("字一色", 13)]
            res.append({"han": sum(h for _, h in yaku), "fu": 25, "yaku": yaku,
                        "yakuman": yaku[0][1] == 13})
            honors = any(t >= 27 for t in pairs)
            suits = {t // 9 for t in pairs if t < 27}
            extra = []
            if all(t not in YAOCHU for t in pairs):
                extra.append(("断幺九", 1))
            if all(t in YAOCHU for t in pairs):
                extra.append(("混老頭", 2))
            if len(suits) == 1:
                extra.append(("混一色", 3) if honors else ("清一色", 6))
            if extra and not res[-1]["yakuman"]:
                res[-1]["yaku"] += extra
                res[-1]["han"] += sum(h for _, h in extra)
        if sum(closed[t] for t in YAOCHU) == 14 and all(closed[t] >= 1 for t in YAOCHU):
            res.append({"han": 13, "fu": 30, "yaku": [("国士無双", 13)], "yakuman": True})
        for s in range(3):
            seg = closed[s * 9:s * 9 + 9]
            if sum(seg) == 14 and seg[0] >= 3 and seg[8] >= 3 and all(seg[k] >= 1 for k in range(1, 8)):
                res.append({"han": 13, "fu": 30, "yaku": [("九蓮宝燈", 13)], "yakuman": True})
    return res


def evaluate(closed, melds, win, tsumo, round_wind=0, seat_wind=1, riichi=False,
             dora=0, ura=0, kuitan=True):
    """和了形の点数を返す。closed は和了牌を含む門前部分(34種の枚数)。
    役が無ければ None。戻り値: dict(han, fu, yaku, yakuman, ron, tsumo_each, tsumo_total, dealer, han_dora)"""
    melds = tuple(melds)
    closed = tuple(int(v) for v in closed)
    return _evaluate(closed, melds, win, bool(tsumo), round_wind, seat_wind, bool(riichi),
                     int(dora), int(ura), bool(kuitan))


@lru_cache(maxsize=200000)
def _evaluate(closed, melds, win, tsumo, round_wind, seat_wind, riichi, dora, ura, kuitan):
    cands = list(_special(closed, melds, win, tsumo, riichi, round_wind, seat_wind))
    for groups, pair in _decomps(closed):
        for wait, wgi in _wait_options(groups, pair, win):
            r = _score_decomp(groups, pair, wait, wgi, melds, win, tsumo, round_wind, seat_wind,
                              riichi, kuitan)
            if r:
                cands.append(r)
    if not cands:
        return None
    dealer = seat_wind == 0
    best = None
    for r in cands:
        r = dict(r)
        r["han_base"] = r["han"]
        if not r["yakuman"]:
            r["han"] += dora + (ura if riichi else 0)
        pay = payments(r["han"], r["fu"], dealer)
        r.update(pay)
        r["dealer"] = dealer
        key = (pay["ron"] if not tsumo else pay["tsumo_total"], r["han"])
        if best is None or key > best[0]:
            best = (key, r)
    return best[1]


# ---------------------------------------------------------------
# 聴牌の手の点数
# ---------------------------------------------------------------
def tenpai_values(closed13, melds, waits, live, round_wind, seat_wind, dora=0, kuitan=True,
                  riichi_possible=True, with_ura=False):
    """聴牌の手(和了牌を含まない門前部分13枚相当)の、待ち牌ごとの点数。

    戻り値 dict:
      dama  : ダマ(リーチなし)の期待値。
          yaku_p  … ロンで役がある待ち牌の割合(残り枚数で重み付け)
          ron     … ロン和了の平均点(役なしの待ちは0点として平均)
          ron_if  … 役がある待ちだけの平均点
          tsumo   … ツモ和了の平均合計点(門前なら必ず役がある)
          han     … ロン和了の平均翻(役ありの待ちのみ)
      riichi: リーチした場合の ron / tsumo / han(裏ドラ・一発は含めない)
              with_ura=True なら ron_x / tsumo_x も入る(裏ドラ・一発で平均0.5翻ぶん上がるとした期待値)
      max_ron : ダマのロン最高点
    副露がある(明かり)手のときは riichi は None。"""
    menzen = not any(m[2] for m in melds)
    tot = sum(live_w for live_w in live.values()) if isinstance(live, dict) else 0
    out = {"waits": []}
    if tot <= 0:
        return None
    acc = {k: 0.0 for k in ("yaku_p", "ron", "ron_if", "ron_if_w", "tsumo", "han", "r_ron", "r_tsumo", "r_han",
                            "r_ron_x", "r_tsumo_x")}
    max_ron = 0
    for w in waits:
        wt = live.get(w, 0)
        if wt <= 0:
            continue
        c = list(closed13)
        c[w] += 1
        d_ron = evaluate(c, melds, w, False, round_wind, seat_wind, False, dora, 0, kuitan)
        d_tsu = evaluate(c, melds, w, True, round_wind, seat_wind, False, dora, 0, kuitan)
        r_ron = evaluate(c, melds, w, False, round_wind, seat_wind, True, dora, 0, kuitan) if menzen else None
        r_tsu = evaluate(c, melds, w, True, round_wind, seat_wind, True, dora, 0, kuitan) if menzen else None
        out["waits"].append({"tile": w, "live": wt, "ron": d_ron, "tsumo": d_tsu,
                             "r_ron": r_ron, "r_tsumo": r_tsu})
        if d_ron:
            acc["yaku_p"] += wt
            acc["ron"] += wt * d_ron["ron"]
            acc["ron_if"] += wt * d_ron["ron"]
            acc["ron_if_w"] += wt
            acc["han"] += wt * d_ron["han"]
            max_ron = max(max_ron, d_ron["ron"])
        if d_tsu:
            acc["tsumo"] += wt * d_tsu["tsumo_total"]
        if r_ron:
            acc["r_ron"] += wt * r_ron["ron"]
            acc["r_han"] += wt * r_ron["han"]
        if r_tsu:
            acc["r_tsumo"] += wt * r_tsu["tsumo_total"]
        if with_ura and r_ron and r_tsu:
            r1 = evaluate(c, melds, w, False, round_wind, seat_wind, True, dora, 1, kuitan)
            t1 = evaluate(c, melds, w, True, round_wind, seat_wind, True, dora, 1, kuitan)
            acc["r_ron_x"] += wt * 0.5 * (r_ron["ron"] + r1["ron"])
            acc["r_tsumo_x"] += wt * 0.5 * (r_tsu["tsumo_total"] + t1["tsumo_total"])
    out["dama"] = {"yaku_p": acc["yaku_p"] / tot, "ron": acc["ron"] / tot,
                   "ron_if": acc["ron_if"] / acc["ron_if_w"] if acc["ron_if_w"] else 0.0,
                   "tsumo": acc["tsumo"] / tot,
                   "han": acc["han"] / acc["ron_if_w"] if acc["ron_if_w"] else 0.0}
    out["riichi"] = ({"ron": acc["r_ron"] / tot, "tsumo": acc["r_tsumo"] / tot, "han": acc["r_han"] / tot,
                      "ron_x": acc["r_ron_x"] / tot, "tsumo_x": acc["r_tsumo_x"] / tot}
                     if menzen and riichi_possible else None)
    out["max_ron"] = max_ron
    return out


def describe(r):
    """点数の結果を '3翻30符 3,900点(タンヤオ・平和)' のような文字列にする。"""
    names = "・".join(n for n, _ in r["yaku"])
    if r["yakuman"]:
        return f"役満 {r['ron']:,}点({names})"
    lv = level_name(r["han"], r["fu"])
    head = lv if lv else f"{r['han']}翻{r['fu']}符"
    return f"{head} ロン{r['ron']:,}点({names})"
