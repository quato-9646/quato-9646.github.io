"""ルールで計算する部品: シャンテン数と、リーチ者に対する牌の安全度。

AIの出力に「理由」として添える数値を計算する。AI自身の判断ではなく、
麻雀の定石をそのまま数式にしたもの。
"""
from functools import lru_cache

SEAT_NAMES = ["自分", "下家", "対面", "上家"]   # 相対席(mjcommon.py と同じ)

# ---------------------------------------------------------------
# シャンテン数(あと何枚で聴牌か。0=聴牌、-1=和了)
# ---------------------------------------------------------------
_YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]  # 幺九牌(国士無双に使う13種)


@lru_cache(maxsize=400000)
def _group_options(counts, is_suit):
    """1グループ(数牌1色の9種、または字牌7種)から作れる (面子数, 搭子数, 雀頭の有無) の組を、
    他に負けない(支配されない)ものだけ集めて返す。同じ形は覚えておいて、2回目からは計算しない。"""
    c = list(counts)
    n = len(c)
    out = set()

    def dfs(i, m, t, p):
        # m=面子の数 t=搭子(あと1枚で面子になる形)の数 p=雀頭があるか
        while i < n and c[i] == 0:
            i += 1
        if i == n:
            out.add((m, t, p))
            return
        if c[i] >= 3:                                   # 刻子
            c[i] -= 3
            dfs(i, m + 1, t, p)
            c[i] += 3
        if is_suit and i + 2 < n and c[i + 1] and c[i + 2]:   # 順子
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            dfs(i, m + 1, t, p)
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        if c[i] >= 2:
            if not p:                                   # 雀頭
                c[i] -= 2
                dfs(i, m, t, 1)
                c[i] += 2
            c[i] -= 2                                   # 対子(搭子として)
            dfs(i, m, t + 1, p)
            c[i] += 2
        if is_suit:
            if i + 1 < n and c[i + 1]:                  # 両面・辺張
                c[i] -= 1; c[i + 1] -= 1
                dfs(i, m, t + 1, p)
                c[i] += 1; c[i + 1] += 1
            if i + 2 < n and c[i + 2]:                  # 嵌張
                c[i] -= 1; c[i + 2] -= 1
                dfs(i, m, t + 1, p)
                c[i] += 1; c[i + 2] += 1
        c[i] -= 1                                       # この1枚はどこにも使わない
        dfs(i, m, t, p)
        c[i] += 1

    dfs(0, 0, 0, 0)
    return _pareto(out)


def _pareto(states):
    """(面子数, 搭子数, 雀頭) の組のうち、同じ雀頭の有無で他より劣るものを除く。"""
    keep = []
    for a in states:
        if not any(b != a and b[2] == a[2] and b[0] >= a[0] and b[1] >= a[1] for b in states):
            keep.append(a)
    return frozenset(keep)


def _merge(a, b):
    """2つのグループの選び方を合わせる(雀頭は全体で1つまで)。"""
    return _pareto({(m1 + m2, t1 + t2, p1 + p2)
                    for m1, t1, p1 in a for m2, t2, p2 in b if p1 + p2 <= 1})


def _shanten_regular(c, n_melds):
    """通常の形(4面子1雀頭)のシャンテン数。c は34種の枚数。"""
    states = _group_options(tuple(c[0:9]), True)
    for lo in (9, 18):
        states = _merge(states, _group_options(tuple(c[lo:lo + 9]), True))
    states = _merge(states, _group_options(tuple(c[27:34]), False))
    best = 8
    for m, t, p in states:
        mm = m + n_melds
        best = min(best, 8 - 2 * mm - min(t, 4 - mm) - p)   # 面子+搭子は4つまでしか数えない
    return best


def shanten(counts, n_melds=0):
    """手牌(34種の枚数)のシャンテン数。副露が無ければ七対子・国士無双も考える。
    手牌の枚数は 13 - 3×副露数(打牌後)か、14 - 3×副露数(和了判定)。"""
    c = [int(v) for v in counts]
    best = _shanten_regular(c, n_melds)
    if n_melds == 0:
        pairs = sum(v >= 2 for v in c)
        kinds = sum(v >= 1 for v in c)
        best = min(best, 6 - pairs + max(0, 7 - kinds))                      # 七対子
        yao = sum(c[t] >= 1 for t in _YAOCHU)
        yao_pair = any(c[t] >= 2 for t in _YAOCHU)
        best = min(best, 13 - yao - (1 if yao_pair else 0))                  # 国士無双
    return best


def discard_shantens(counts, n_melds=0):
    """手牌から各牌を1枚切った後のシャンテン数を {牌の種類: 向聴数} で返す。"""
    c = [int(v) for v in counts]
    out = {}
    for t in range(34):
        if c[t] > 0:
            c[t] -= 1
            out[t] = shanten(c, n_melds)
            c[t] += 1
    return out


def shanten_text(s):
    return "和了形" if s < 0 else "聴牌" if s == 0 else f"{s}向聴"


# ---------------------------------------------------------------
# 安全度(リーチ者に切って当たりにくいか)
# ---------------------------------------------------------------
LEVEL_NAMES = ["現物", "スジ", "字牌(安全寄り)", "半スジ", "字牌", "無筋(端牌)", "無筋(中張牌)"]
SAFE_MAX_LEVEL = 2   # この番号以下を「安全牌」とみなす(現物・スジ・字牌(安全寄り))

# 数牌 n(0始まり: 0=1, 8=9) の筋になる牌。4・5・6 は両方切られて初めて「スジ」
_SUJI = {0: (3,), 1: (4,), 2: (5,), 3: (0, 6), 4: (1, 7), 5: (2, 8), 6: (3,), 7: (4,), 8: (5,)}


def tile_level(t, x, riichi_seats, visible):
    """牌 t を、リーチ中の相手(相対席のリスト)全員に切るときの安全度の段階。
    段階が大きいほど危険。複数人がリーチ中なら、一番危険な段階を返す。
        x       : 局面の牌の表(mjcommon.py の x。捨て牌と現物を使う)
        visible : 場に見えている枚数(34種)"""
    worst = 0
    for k in riichi_seats:
        if x[9 + k][t] > 0:                       # 現物
            lv = 0
        elif t >= 27:                             # 字牌
            lv = 2 if visible[t] >= 3 else 4
        else:
            n, base = t % 9, t - t % 9
            hit = sum(1 for s in _SUJI[n] if x[1 + k][base + s] > 0)
            if hit == len(_SUJI[n]):
                lv = 1                            # スジ
            elif hit > 0:
                lv = 3                            # 半スジ(4・5・6の片側のスジ)
            else:
                lv = 5 if n in (0, 8) else 6      # 無筋
        worst = max(worst, lv)
    return worst


def riichi_seats(f):
    """局面の数値 f から、リーチ中の相手(相対席 1〜3)のリストを返す。"""
    return [k for k in range(1, 4) if f[k]]


def seats_text(seats):
    return "・".join(SEAT_NAMES[k] for k in seats) + "リーチ"
