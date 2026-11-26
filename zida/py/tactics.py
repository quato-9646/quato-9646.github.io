"""定石(ワンチャンス・ノーチャンス)の判定。学習モデルとは別に、見えている牌の枚数から数えて言葉にする。
リーチ者の両面待ちが成立するために必要な牌が、場にどれだけ見えているかを見る。"""
from mjcommon import NAMES

# 数牌 n(0始まり)を当たり牌にする両面待ちの形: 手の中の2枚が (n-2,n-1) なら n と n-3 待ち、(n+1,n+2) なら n と n+3 待ち
_SIDES = ((-2, -1), (1, 2))


def _vis(x):
    v = x[0] + x[1] + x[2] + x[3] + x[4] + x[6] + x[7] + x[8] + x[9]
    return [int(a) for a in v]


def chance(ev, t):
    """リーチ中の相手に t を切るときの、両面待ち・字牌の残り枚数から見た注意点を返す。なければ None。"""
    x, f = ev["x"], ev["f"]
    if not any(f[k] for k in range(1, 4)):
        return None
    vis = _vis(x)
    nm = NAMES[t]
    if t >= 27:
        if vis[t] >= 4:
            return None
        if vis[t] >= 3:
            return f"{nm}は3枚見えで、残り1枚。シャンポン待ちは成立せず、当たるのはタンキ待ちだけなので、かなり安全な牌です(ワンチャンス)。"
        return None
    n, base = t % 9, t - t % 9
    rest = []                                  # 両面待ちが成立しうる側の、必要牌の残り枚数
    for a, b in _SIDES:
        if not (0 <= n + a < 9 and 0 <= n + b < 9):
            continue
        rest.append(min(4 - vis[base + n + a], 4 - vis[base + n + b]))
    if not rest:
        return None
    if max(rest) <= 0:
        return (f"{nm}は、両面待ちに必要な牌が場に4枚見えていて、両面は成立しません(ノーチャンス)。"
                "当たるのはカンチャン・シャンポン・タンキだけなので、無筋でもかなり安全寄りです。")
    if max(rest) == 1:
        return (f"{nm}を待つ両面は、必要な牌が残り1枚で、成立しにくい形です(ワンチャンス)。"
                "無筋でも、普通の無筋より当たりにくい牌です。")
    return None


# ---------------------------------------------------------------
# コンボ理論: 牌 t が当たる待ちの形を、残り枚数の組み合わせ数(ポイント)で数える
# ---------------------------------------------------------------
def _seen(x):
    """見えている枚数(自分の手牌・全員の河・副露・ドラ表示牌)。"""
    v = _vis(x)
    for d in range(34):
        c = int(x[5][d])
        if c > 0:                              # ドラ牌 d の表示牌は、1つ前の牌
            n = d % 9 if d < 27 else None
            if d < 27:
                ind = d - 1 if n > 0 else d + 8
            elif d < 31:
                ind = d - 1 if d > 27 else 30
            else:
                ind = d - 1 if d > 31 else 33
            v[ind] += min(c, 1)
    return v


def combo_points(ev, t, k):
    """リーチ者(相対席 k)に t を切ったときの当たりの組み合わせ数。
    返り値: (合計, {"両面":n, "嵌張":n, "辺張":n, "シャンポン":n})。現物は (0, {})。単騎は数えない。"""
    x = ev["x"]
    if x[9 + k][t] > 0:
        return 0, {}
    seen = _seen(x)
    rem = lambda u: max(0, 4 - seen[u])
    river = x[1 + k]
    parts = {"両面": 0, "嵌張": 0, "辺張": 0, "シャンポン": 0}
    parts["シャンポン"] = rem(t) * (rem(t) - 1) // 2      # 手の中に t の対子を持つ組み合わせ
    if t >= 27:
        return parts["シャンポン"], {a: b for a, b in parts.items() if b}
    n = t % 9
    # 下側: 手の中に (t-2, t-1) を持つ形
    if n >= 2:
        p = rem(t - 2) * rem(t - 1)
        if n == 2:
            parts["辺張"] += p                  # 1・2 で 3 待ち
        elif river[t - 3] == 0:                 # もう片方の待ち牌を切っていれば振聴で当たらない
            parts["両面"] += p
    # 上側: 手の中に (t+1, t+2) を持つ形
    if n <= 6:
        p = rem(t + 1) * rem(t + 2)
        if n == 6:
            parts["辺張"] += p                  # 8・9 で 7 待ち
        elif river[t + 3] == 0:
            parts["両面"] += p
    # 嵌張: (t-1, t+1)
    if 1 <= n <= 7:
        parts["嵌張"] += rem(t - 1) * rem(t + 1)
    parts = {a: b for a, b in parts.items() if b}
    return sum(parts.values()), parts


def combo_text(ev, t):
    """tを切ったときのコンボ理論のポイントを説明する文。リーチ者がいなければ None。複数なら最も高い人。"""
    f = ev["f"]
    seats = [k for k in range(1, 4) if f[k]]
    if not seats:
        return None
    best = None
    for k in seats:
        pts, parts = combo_points(ev, t, k)
        if best is None or pts > best[0]:
            best = (pts, parts, k)
    pts, parts, k = best
    nm = NAMES[t]
    if ev["x"][9 + k][t] > 0:
        return f"{nm}はリーチ者の現物で、当たりは0ポイントです。"
    if pts == 0:
        return f"コンボ理論では、{nm}は当たる組み合わせが残っておらず、0ポイントです(単騎を除く)。"
    detail = "・".join(f"{a}{b}" for a, b in parts.items()) or "なし"
    return f"コンボ理論では、{nm}の当たりは{pts}ポイント({detail})です。"


def combo_rank(ev, t):
    """手牌の中で、tが何番目に危険か(ポイントの高い順)と、手牌の最大・最小ポイントを返す。"""
    x, f = ev["x"], ev["f"]
    seats = [k for k in range(1, 4) if f[k]]
    if not seats:
        return None
    tiles = [u for u in range(34) if x[0][u] > 0]
    if t not in tiles:
        return None
    score = {u: max(combo_points(ev, u, k)[0] for k in seats) for u in tiles}
    order = sorted(tiles, key=lambda u: -score[u])
    return order.index(t) + 1, len(tiles), score[t], min(score.values()), max(score.values())
