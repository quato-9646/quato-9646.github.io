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
