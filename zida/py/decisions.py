"""リーチ判断・鳴き判断のための部品: 判断場面の取り出しと、モデルの形。

牌譜を1つずつ読み、次の2種類の「判断の場面」を正解ラベル付きで取り出す。

  ・リーチ判断 … 門前で聴牌できる手のとき、リーチする(1)か、ダマ(リーチせず打牌)する(0)か
  ・鳴き判断   … 他家の打牌に対して鳴ける手のとき、パス(0)・チー(1)・ポン(2)・大明槓(3)のどれか

局面の表し方は mjcommon.py の x(牌の表)と f(数値の列)に、g(場況の数値の列)を足したもの。
mjcommon.parse_game(打牌の予測)とは別に、場の状態を追いかけ直している。
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:          # ブラウザ内(Pyodide)などtorchが無い環境では、numpy版(npnet.py)で推論する
    torch = nn = None
_Module = nn.Module if nn is not None else object

from mjcommon import (BASE_YEAR, DEFAULT_YEAR, DISCARD_RE, DRAW_RE, NUM_CH, NUM_F, WINDS,
                      decode_meld, dora_from_indicator)
from rules import discard_shantens, shanten
from scoring import evaluate, tenpai_values

FIVES = (4, 13, 22)       # 5萬・5筒・5索の牌の種類番号
NUM_V = 12               # リーチ判断の打点の数値の列(riichi_event の V)
AKA_IDS = (16, 52, 88)   # 赤5萬・赤5筒・赤5索の牌番号
YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]   # 1・9・字牌
CALL_PASS, CALL_CHI, CALL_PON, CALL_KAN = 0, 1, 2, 3
CALL_NAMES = ["パス", "チー", "ポン", "大明槓"]

# g(場況の数値の列)の中身。riichi/call のうち使わない欄は0のまま。
#    0〜3  : 持ち点(自分・下家・対面・上家)。1000点単位
#    4     : 場風(0=東 1=南 2=西)    5: 自風(自分が親から数えて何番目か。0=親)
#    6     : 本場                    7: 場に出ているリーチ棒の数
#    8     : 山の残り枚数            9: 手牌にあるドラの枚数
#    10    : 手牌にある1・9・字牌の枚数  11: 自分の副露(鳴き)の数
#    12    : 手牌に対子以上ある役牌(白發中・場風・自風)の種類数   13: 順位(1〜4)
#   --- リーチ判断だけ ---
#    14    : 最良の聴牌の待ち牌の残り枚数    15: 聴牌になる切り方の数
#   --- 鳴き判断だけ ---
#    16    : 今の向聴数+2         17: ポンした後の向聴数+2(できなければ0)
#    18    : チーした後の向聴数+2(最良。できなければ0)   19: 大明槓した後の向聴数+2
#    20    : 切られた牌が役牌なら 枚数(0〜2。場風かつ自風なら2)
#    21    : 切られた牌のドラ枚数    22: 切られた牌が1・9・字牌か
#    23    : 切った人が自分から見て何番目か(1=下家 2=対面 3=上家)
#    24    : 自分の副露にある役牌の翻数    25: 手牌+副露が全て中張牌か(断幺九が残っているか。喰いタンありのとき)
#    26    : 一番多い色+字牌の外にある牌の枚数(0なら混一色の形)   27: 字牌の枚数
#    28    : 手牌の対子以上の種類数(対々和・七対子の傾向)       29: 手牌+副露のドラ枚数(赤含む)
#    30    : 赤ドラの枚数                                      31: 切られた牌が中張牌で、断幺九が残っているか
NUM_G = 32
G_SCALE = np.array([10, 10, 10, 10, 3, 3, 4, 4, 70, 4, 14, 4, 3, 4,
                    20, 5, 10, 10, 10, 10, 2, 3, 1, 3,
                    3, 1, 13, 14, 7, 5, 3, 1], dtype=np.float32)


# ---------------------------------------------------------------
# 待ち牌(聴牌の手があと何で和了れるか)
# ---------------------------------------------------------------
def tenpai_waits(c, n_melds, visible):
    """聴牌の手牌 c(34種の枚数のリスト。変更しても元に戻す)の、待ち牌と残り枚数を返す。
    visible は場に見えている枚数(自分の手牌を含む)。戻り値: (待ち牌のリスト, 残り枚数の合計)"""
    cand = set()
    for u in range(34):                     # 手牌の牌と、その前後2つ以内の同じ色の牌が候補
        if c[u]:
            cand.add(u)
            if u < 27:
                base = u - u % 9
                for d in (-2, -1, 1, 2):
                    if base <= u + d < base + 9:
                        cand.add(u + d)
    if n_melds == 0 and sum(1 for u in YAOCHU if c[u]) >= 10:   # 国士無双の形
        cand.update(YAOCHU)
    waits = []
    for w in sorted(cand):
        if c[w] >= 4:
            continue
        c[w] += 1
        if shanten(c, n_melds) < 0:         # 1枚足して和了形になる牌
            waits.append(w)
        c[w] -= 1
    live = sum(max(4 - int(visible[w]), 0) for w in waits)
    return waits, live


def call_type(m):
    """副露コードから、他家の打牌への鳴きの種類を返す(自分の番の槓・北抜きはNone)。"""
    if m & 0x4:
        return CALL_CHI
    if m & 0x8:
        return CALL_PON
    if m & 0x10 or m & 0x20:
        return None
    return CALL_KAN if (m & 3) != 0 else None


# ---------------------------------------------------------------
# 1局の場の状態
# ---------------------------------------------------------------
class _Round:
    """1局ぶんの場の状態。牌譜のタグを1つずつ読みながら更新し、判断場面の特徴量を作る。"""

    def __init__(self, el, year, rules=None, full=False):
        rules = rules or {}
        self.aka_on = rules.get("aka", True)           # 赤ドラあり
        self.kuitan = rules.get("kuitan", True)        # 喰いタンあり
        self.last_round = rules.get("last_round", 7)   # 最終局の番号(東1=0 … 南4=7)
        self.calls = rules.get("calls", True)          # Falseなら鳴き判断の場面を作らない(リーチ判断だけ欲しいとき)
        self.full = full                               # Trueなら、聴牌の各切り方の打点の詳細も残す
        seed = [int(v) for v in el.get("seed").split(",")]
        self.round_idx = seed[0]
        self.new_sticks = 0                            # この局の中で増えたリーチ棒(特徴量の g[7] には含めない)
        self.label = f"{WINDS[seed[0] // 4]}{seed[0] % 4 + 1}局{seed[1]}本場"
        self.round_wind, self.honba, self.sticks = seed[0] // 4, seed[1], seed[2]
        self.oya = int(el.get("oya"))
        self.scores = [int(v) for v in el.get("ten").split(",")]
        self.year_offset = min(max(year - BASE_YEAR, 0), 255)
        self.hand = np.zeros((4, 34), dtype=np.int8)
        self.aka_h = np.zeros((4, 3), dtype=np.int8)   # 手牌にある赤5(萬・筒・索)
        self.aka_m = [0] * 4                           # 副露にある赤5の枚数
        self.meld_list = [[] for _ in range(4)]        # 副露の詳細 (種類, 牌, 明か)
        self.last_discard_id = None
        for i in range(4):
            for t in el.get(f"hai{i}").split(","):
                self.hand[i, int(t) // 4] += 1
                if int(t) in AKA_IDS:
                    self.aka_h[i, AKA_IDS.index(int(t))] += 1
        self.discards = np.zeros((4, 34), dtype=np.int8)
        self.meld_tiles = np.zeros((4, 34), dtype=np.int8)
        self.genbutsu = np.zeros((4, 34), dtype=np.int8)
        self.indicators = np.zeros(34, dtype=np.int8)
        self.called = np.zeros(34, dtype=np.int8)
        self.dora = np.zeros(34, dtype=np.int8)
        self.dora[dora_from_indicator(seed[5] // 4)] += 1
        self.indicators[seed[5] // 4] += 1
        self.melds = [0] * 4        # 副露の数(暗槓を含む)
        self.n_open = [0] * 4       # 明かりの副露の数(チー・ポン・大明槓。暗槓は含まない)
        self.junme = [0] * 4
        self.n_draws = 0            # ここまでのツモの回数(嶺上牌を含む)
        self.last_discard = None
        self.pending_riichi = [False] * 4
        self.in_riichi = [False] * 4
        self.broken = False

    # ---- 赤ドラ・副露・打点の部品 ----
    def add_tile(self, p, tid):
        if tid in AKA_IDS:
            self.aka_h[p, AKA_IDS.index(tid)] += 1

    def remove_tile(self, p, tid):
        if tid in AKA_IDS:
            self.aka_h[p, AKA_IDS.index(tid)] -= 1

    def record_meld(self, who, m):
        """副露の詳細(種類・牌・赤ドラ)を記録する。手牌の枚数の更新は呼び出し側で行う。"""
        called = self.last_discard_id
        if m & 0x4:                                   # チー
            t = m >> 10
            r = t % 3
            t //= 3
            base = (t // 7) * 9 + t % 7
            subs = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
            ids = [(base + i) * 4 + subs[i] for i in range(3)]
            self.meld_list[who].append(("chi", base, True))
        elif m & 0x8:                                 # ポン
            kind = (m >> 9) // 3
            unused = (m >> 5) & 3
            ids = [kind * 4 + k for k in range(4) if k != unused]
            self.meld_list[who].append(("pon", kind, True))
        elif m & 0x10:                                # 加槓
            kind = (m >> 9) // 3
            tid = kind * 4 + ((m >> 5) & 3)
            self.remove_tile(who, tid)
            self.aka_m[who] += tid in AKA_IDS
            lst = self.meld_list[who]
            for i, (k, t, o) in enumerate(lst):
                if k == "pon" and t == kind:
                    lst[i] = ("kan", kind, True)
            return
        elif m & 0x20:                                # 北抜き
            return
        else:                                         # 大明槓・暗槓
            kind = (m >> 8) // 4
            ids = [kind * 4 + k for k in range(4)]
            is_open = (m & 3) != 0
            self.meld_list[who].append(("kan", kind, is_open))
            if not is_open:
                called = None
        for tid in ids:
            if tid != called:
                self.remove_tile(who, tid)
        self.aka_m[who] += sum(1 for tid in ids if tid in AKA_IDS)

    def dora_total(self, p, hand=None, aka_h=None):
        """手牌+副露にあるドラ(赤ドラ含む)の枚数。hand/aka_h を渡せば、その手牌で数える。"""
        h = self.hand[p] if hand is None else hand
        a = self.aka_h[p] if aka_h is None else aka_h
        n = int(((h + self.meld_tiles[p]) * self.dora).sum()) + self.aka_m[p]
        if self.aka_on:
            n += int(a.sum())
        return n

    def ctx(self, p):
        """点数状況の判断に使う、局の情報。"""
        return {"round_idx": self.round_idx, "last_round": self.last_round, "honba": self.honba,
                "sticks": self.sticks + self.new_sticks, "scores": [s * 100 for s in self.scores], "me": p,
                "oya": self.oya, "seat": (p - self.oya) % 4, "round_wind": self.round_wind,
                "draws_left": max(70 - self.n_draws, 0)}

    # ---- 特徴量 ----
    def visible(self, p):
        return (self.hand[p].astype(np.int16) + self.discards.sum(0) - self.called
                + self.meld_tiles.sum(0) + self.indicators)

    def x_base(self, p):
        x = np.zeros((NUM_CH, 34), dtype=np.uint8)
        x[0] = self.hand[p]
        for k in range(4):
            x[1 + k] = self.discards[(p + k) % 4]
            x[6 + k] = self.meld_tiles[(p + k) % 4]
        x[5] = self.dora
        for k in range(1, 4):
            x[9 + k] = self.genbutsu[(p + k) % 4]
        return x

    def f_base(self, p, shanten_plus1):
        f = np.zeros(NUM_F, dtype=np.uint8)   # f[0] は「このリーチ宣言」なので答えを教えないよう0のまま
        for k in range(1, 4):
            f[k] = self.in_riichi[(p + k) % 4]
        for k in range(4):
            f[4 + k] = (p + k) % 4 == self.oya
        f[8] = min(self.junme[p], 255)
        f[9] = self.year_offset
        f[10] = shanten_plus1
        return f

    def g_base(self, p):
        g = np.zeros(NUM_G, dtype=np.uint8)
        for k in range(4):
            g[k] = min(max(self.scores[(p + k) % 4] // 10, 0), 255)
        seat = (p - self.oya) % 4
        g[4], g[5] = self.round_wind, seat
        g[6], g[7] = min(self.honba, 255), min(self.sticks, 255)
        g[8] = max(70 - self.n_draws, 0)
        h = self.hand[p]
        g[9] = int((h * self.dora).sum())
        g[10] = int(h[YAOCHU].sum())
        g[11] = self.n_open[p]
        yakuhai = {31, 32, 33, 27 + self.round_wind, 27 + seat}
        g[12] = sum(1 for t in yakuhai if h[t] >= 2)
        g[13] = 1 + sum(1 for q in range(4) if self.scores[q] > self.scores[p])
        return g

    # ---- リーチ判断の場面 ----
    def riichi_event(self, who, declared):
        """who が打牌する直前の局面がリーチ判断の場面なら、その特徴量とラベルを返す。"""
        hand = self.hand[who]
        sh = discard_shantens(hand, self.melds[who])
        if not sh:
            return None
        best = min(sh.values())
        eligible = (self.n_open[who] == 0 and best == 0 and self.scores[who] >= 10
                    and 70 - self.n_draws >= 4)
        if not (eligible or declared):     # リーチした手は、条件の判定がずれても必ず入れる
            return None
        x = self.x_base(who)
        for t, v in sh.items():
            x[14][t] = v + 1
        vis = self.visible(who)
        waits_by_t = {}
        for t, v in sh.items():
            if v == 0:                       # 切ると聴牌になる牌 → 待ち牌と残り枚数
                c = [int(v2) for v2 in hand]
                c[t] -= 1
                waits, live = tenpai_waits(c, self.melds[who], vis)
                waits_by_t[t] = (waits, live)
                x[13][t] = min(live, 12)
        g = self.g_base(who)
        best_t, live_max = None, 0
        if waits_by_t:
            best_t = max(waits_by_t, key=lambda t: waits_by_t[t][1])
            live_max = waits_by_t[best_t][1]
        g[14] = min(live_max, 255)
        g[15] = len(waits_by_t)

        # ---- 打点の見積もり(待ち牌ごとに役・符・翻を計算) ----
        values = {}
        seat = (who - self.oya) % 4
        melds = tuple(self.meld_list[who])
        for t, (waits, live) in waits_by_t.items():
            c13 = [int(v) for v in hand]
            c13[t] -= 1
            h_after = hand.copy()
            h_after[t] -= 1
            aka_after = self.aka_h[who].copy()
            if t in FIVES and h_after[t] == 0:
                aka_after[FIVES.index(t)] = 0
            dora = self.dora_total(who, h_after, aka_after)
            live_w = {w: max(4 - int(vis[w]), 0) for w in waits}
            tv = tenpai_values(c13, melds, waits, live_w, self.round_wind, seat, dora, self.kuitan, with_ura=self.full)
            if tv:
                tv["dora"] = dora
            values[t] = tv
        V = np.zeros(NUM_V, dtype=np.float32)
        cap = lambda v: min(v / 1000.0, 48.0)
        if best_t is not None and values.get(best_t):
            tv = values[best_t]
            d, r = tv["dama"], tv["riichi"] or {"ron": 0.0, "tsumo": 0.0, "han": 0.0}
            V[:8] = [d["yaku_p"], cap(d["ron"]), cap(d["ron_if"]), cap(d["tsumo"]),
                     cap(r["ron"]), cap(r["tsumo"]), d["han"], r["han"]]
            V[11] = cap(tv["max_ron"])
            V[10] = tv["dora"]
        good = [tv for tv in values.values() if tv]
        if good:
            V[8] = max(cap(tv["dama"]["ron"]) for tv in good)
            V[9] = sum(tv["dama"]["yaku_p"] > 0.5 for tv in good)
        ev = {
            "who": who, "junme": self.junme[who], "x": x, "f": self.f_base(who, best + 1), "g": g, "V": V,
            "label": int(declared), "eligible": eligible,
            "best_t": best_t, "waits": waits_by_t[best_t][0] if best_t is not None else [],
            "live": live_max, "shanten": best, "ctx": self.ctx(who),
            "tenpai_ts": list(waits_by_t.keys()),     # 切ると聴牌になる牌
        }
        if self.full:
            ev["values"] = values
            ev["live_by_t"] = {t: wl for t, wl in waits_by_t.items()}
        return ev

    # ---- 鳴き判断の場面 ----
    def call_opportunities(self, d, t):
        """d が牌 t を切った直後に、他家が鳴ける場面の一覧(ラベルは後で確定)。"""
        opps = []
        if not self.calls:
            return opps
        if 70 - self.n_draws <= 0:           # 河底(最後の打牌)は鳴けない
            return opps
        for p in range(4):
            if p == d or self.in_riichi[p]:
                continue
            h = self.hand[p]
            legal = [True, False, False, False]
            legal[CALL_PON] = bool(h[t] >= 2)
            legal[CALL_KAN] = bool(h[t] >= 3)
            chi_sets = []
            if p == (d + 1) % 4 and t < 27:      # チーできるのは、切った人の下家だけ
                n = t % 9
                for a, b, ok in ((t - 2, t - 1, n >= 2), (t - 1, t + 1, 1 <= n <= 7),
                                 (t + 1, t + 2, n <= 6)):
                    if ok and h[a] and h[b]:
                        chi_sets.append((a, b))
                legal[CALL_CHI] = bool(chi_sets)
            if not any(legal[1:]):
                continue
            if h.sum() != 13 - 3 * self.melds[p]:
                self.broken = True
                continue
            c = [int(v) for v in h]
            m = self.melds[p]
            sh_now = shanten(c, m)
            sh_after = [None, None, None, None]
            if legal[CALL_PON]:
                c[t] -= 2
                sh_after[CALL_PON] = shanten(c, m + 1)
                c[t] += 2
            if legal[CALL_KAN]:
                c[t] -= 3
                sh_after[CALL_KAN] = shanten(c, m + 1)
                c[t] += 3
            for a, b in chi_sets:
                c[a] -= 1; c[b] -= 1
                s = shanten(c, m + 1)
                c[a] += 1; c[b] += 1
                if sh_after[CALL_CHI] is None or s < sh_after[CALL_CHI]:
                    sh_after[CALL_CHI] = s

            x = self.x_base(p)
            x[13][t] = 1                        # 切られた牌
            g = self.g_base(p)
            g[16] = sh_now + 2
            for col, k in ((17, CALL_PON), (18, CALL_CHI), (19, CALL_KAN)):
                g[col] = sh_after[k] + 2 if sh_after[k] is not None else 0   # 鳴けないなら0
            seat = (p - self.oya) % 4
            g[20] = (t in (31, 32, 33)) + (t == 27 + self.round_wind) + (t == 27 + seat)
            g[21] = int(self.dora[t])
            g[22] = int(t in YAOCHU)
            g[23] = (d - p) % 4
            tot = h.astype(np.int16) + self.meld_tiles[p]
            seat_p = (p - self.oya) % 4
            for kind_m, tm, _ in self.meld_list[p]:
                if kind_m != "chi":
                    g[24] += int(tm in (31, 32, 33)) + int(tm == 27 + self.round_wind) + int(tm == 27 + seat_p)
            g[25] = int(self.kuitan and tot[YAOCHU].sum() == 0)
            g[26] = min(int(tot.sum() - max(tot[s * 9:s * 9 + 9].sum() + tot[27:].sum() for s in range(3))), 255)
            g[27] = int(tot[27:].sum())
            g[28] = int((h >= 2).sum())
            g[29] = min(self.dora_total(p), 255)
            g[30] = int(self.aka_h[p].sum() + self.aka_m[p]) if self.aka_on else 0
            g[31] = int(g[25] and t not in YAOCHU)
            opps.append({
                "p": p, "d": d, "tile": t, "x": x, "f": self.f_base(p, sh_now + 1), "g": g,
                "legal": legal, "sh_now": sh_now, "sh_after": sh_after, "junme": self.junme[p],
                "ctx": self.ctx(p),
            })
        return opps


def game_rules(root):
    """牌譜の GO タグから、赤ドラの有無・喰いタンの有無・最終局の番号を読む。"""
    rules = {"aka": True, "kuitan": True, "last_round": 7}
    for el in root:
        if el.tag == "GO":
            t = int(el.get("type", "0"))
            rules["aka"] = not (t & 0x02)
            rules["kuitan"] = not (t & 0x04)
            rules["last_round"] = 7 if (t & 0x08) else 3
            break
    return rules


def parse_decisions(root, year=DEFAULT_YEAR, full=False, calls=True):
    """1つの牌譜から、局ごとにリーチ判断・鳴き判断の場面を取り出す。

    戻り値: 局のリスト。各局は dict で
        label, broken(手牌の追跡が合わなくなった局ならTrue)
        riichi_events : riichi_event の dict のリスト(label: 1=リーチ 0=ダマ)
        call_events   : call_opportunities の dict のリスト(label: 0=パス 1=チー 2=ポン 3=大明槓)
        call_missed   : 実際に鳴いたのに、鳴ける場面として検出できなかった回数(確認用。0が正常)
        riichi_called : 牌譜のリーチ宣言の回数(確認用)
    """
    rounds = []
    cur = None
    st = None
    rules = game_rules(root)
    rules["calls"] = calls
    pending = None      # 直前の打牌に対する、鳴き判断の場面(まだラベル未確定)

    def resolve(caller, ctype):
        """直前の打牌への鳴きの結果が分かったので、ラベルを付けて確定する。"""
        for op in pending["opps"]:
            if caller is None:
                label = CALL_PASS
            elif op["p"] == caller:
                label = ctype
            else:
                # 他の人が鳴いた。それが自分の選択肢より優先(ポン・カン > チー)なら選べなかったので除く
                mine = 2 if (op["legal"][CALL_PON] or op["legal"][CALL_KAN]) else 1
                if (2 if ctype >= CALL_PON else 1) > mine:
                    continue
                label = CALL_PASS
            op["label"] = label
            cur["call_events"].append(op)
        if caller is not None and all(op["p"] != caller for op in pending["opps"]):
            cur["call_missed"] += 1

    for el in root:
        tag = el.tag
        if tag == "INIT":
            pending = None
            st = _Round(el, year, rules, full)
            cur = {"label": st.label, "broken": False, "riichi_events": [], "call_events": [],
                   "call_missed": 0, "riichi_called": 0, "results": []}
            rounds.append(cur)
            continue
        if cur is None:
            continue

        if tag == "DORA":
            t = int(el.get("hai")) // 4
            st.dora[dora_from_indicator(t)] += 1
            st.indicators[t] += 1
        elif tag == "N":
            who, m = int(el.get("who")), int(el.get("m"))
            ctype = call_type(m)
            if pending is not None:
                if ctype is not None:
                    resolve(who, ctype)
                else:
                    resolve(None, None)
                pending = None
            elif ctype is not None:
                cur["call_missed"] += 1      # 鳴いたのに、鳴ける場面として検出できていなかった
            st.record_meld(who, m)
            removed, shown, new_meld, was_called = decode_meld(m)
            for t in removed:
                st.hand[who, t] -= 1
            for t in shown:
                st.meld_tiles[who, t] += 1
            if new_meld:
                st.melds[who] += 1
                if was_called:
                    st.n_open[who] += 1
            if was_called and st.last_discard is not None:
                st.called[st.last_discard] += 1
            if st.hand[who].min() < 0:
                st.broken = True
        elif tag == "REACH":
            if el.get("step") == "1":
                st.pending_riichi[int(el.get("who"))] = True
                cur["riichi_called"] += 1
            elif el.get("ten"):
                st.scores = [int(v) for v in el.get("ten").split(",")]   # リーチ棒を出した後の持ち点
                st.new_sticks += 1
        elif tag in ("AGARI", "RYUUKYOKU"):
            pending = None                  # ロンなど。鳴きの結果は分からないので使わない
            sc = el.get("sc")
            delta = [int(v) * 100 for v in sc.split(",")[1::2]] if sc else [0, 0, 0, 0]
            cur["results"].append({"tag": tag, "who": int(el.get("who", -1)),
                                   "from": int(el.get("fromWho", -1)), "delta": delta})
        else:
            m = DRAW_RE.fullmatch(tag)
            if m:
                if pending is not None:
                    resolve(None, None)
                    pending = None
                who = "TUVW".index(m.group(1))
                st.hand[who, int(m.group(2)) // 4] += 1
                st.add_tile(who, int(m.group(2)))
                st.n_draws += 1
                continue
            m = DISCARD_RE.fullmatch(tag)
            if m:
                who = "DEFG".index(m.group(1))
                tid = int(m.group(2))
                t = tid // 4
                st.junme[who] += 1
                if not st.in_riichi[who]:
                    if st.hand[who, t] <= 0 or st.hand[who].sum() != 14 - 3 * st.melds[who]:
                        st.broken = True
                    ev = st.riichi_event(who, st.pending_riichi[who])
                    if ev:
                        ev["actual_t"] = t          # 実際に切った牌
                        ev["kept"] = int(t in ev["tenpai_ts"])   # 聴牌を保つ(or作る)牌を切ったか
                        cur["riichi_events"].append(ev)
                st.hand[who, t] -= 1
                st.remove_tile(who, tid)
                st.discards[who, t] += 1
                st.last_discard = t
                st.last_discard_id = tid
                st.genbutsu[who, t] = 1
                for k in range(4):
                    if k != who and st.in_riichi[k]:
                        st.genbutsu[k, t] = 1
                if st.hand[who].min() < 0:
                    st.broken = True
                opps = st.call_opportunities(who, t)
                pending = {"opps": opps} if opps else None
                if st.pending_riichi[who]:
                    st.in_riichi[who] = True
                    st.pending_riichi[who] = False
        cur["broken"] = st.broken
    if cur is not None:
        cur["broken"] = st.broken
    for r in rounds:
        annotate_outcomes(r)
    return rounds


def annotate_outcomes(r):
    """局の結果(AGARI/RYUUKYOKU)を、その局のリーチ判断の場面に紐づける。
        win     : その人がこの局で和了したか(ツモ・ロン。ダブロンも含む)
        tsumo   : うちツモ和了か
        dealin  : 他家の和了に放銃したか
        delta   : この局での点数の増減(点。本場・供託・流局の罰符を含む)
        pure    : ダマの場面で、この局の後でリーチをかけていないか(リーチの場面は常にTrue)"""
    res = r["results"]
    evs = r["riichi_events"]
    for i, ev in enumerate(evs):
        w = ev["who"]
        wins = [a for a in res if a["tag"] == "AGARI" and a["who"] == w]
        ev["win"] = int(bool(wins))
        ev["tsumo"] = int(any(a["from"] == w for a in wins))
        ev["dealin"] = int(any(a["tag"] == "AGARI" and a["from"] == w and a["who"] != w for a in res))
        ev["delta"] = int(sum(a["delta"][w] for a in res))
        later = any(e2["who"] == w and e2["label"] == 1 for e2 in evs[i + 1:])
        ev["pure"] = int(ev["label"] == 1 or not later)
    for ev in r["call_events"]:           # 鳴き判断の場面にも、その人のこの局の結果を付ける
        w = ev["p"]
        wins = [a for a in res if a["tag"] == "AGARI" and a["who"] == w]
        ev["win"] = int(bool(wins))
        ev["tsumo"] = int(any(a["from"] == w for a in wins))
        ev["dealin"] = int(any(a["tag"] == "AGARI" and a["from"] == w and a["who"] != w for a in res))
        ev["delta"] = int(sum(a["delta"][w] for a in res))


# ---------------------------------------------------------------
# モデル
# ---------------------------------------------------------------
class DecisionNet(_Module):
    """局面 (x: 15×34, f: 11個, g: 24個) → n_out 個の点数 を出す全結合ネットワーク。
    リーチ判断は n_out=1(大きいほどリーチ)、鳴き判断は n_out=4(パス・チー・ポン・大明槓)。"""

    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CH * 34 + NUM_F + NUM_G, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_out),
        )
        self.register_buffer("g_scale", torch.tensor(G_SCALE))

    def forward(self, x, f, g, legal=None):
        f = f.float()
        f = torch.cat([f[:, :8], f[:, 8:9] / 18, f[:, 9:10] / 17, f[:, 10:] / 8], dim=1)
        h = torch.cat([x.float().flatten(1) / 4, f, g.float() / self.g_scale], dim=1)
        out = self.net(h)
        if legal is not None:            # 鳴けない選択肢は選ばれないようにする
            out = out.masked_fill(~legal, -1e9)
        return out


def riichi_features(X, F, G, V):
    """リーチ判断用の数値の列(決定木モデル用)。X,F,G,V は (N,...) の配列(Vは打点の見積もり)。"""
    import numpy as np
    h = X[:, 0]
    extra = np.stack([h[:, 0:9].sum(1), h[:, 9:18].sum(1), h[:, 18:27].sum(1), h[:, 27:].sum(1),
                      (X[:, 13] * (X[:, 14] == 1)).max(1), (X[:, 13] > 0).sum(1)], 1)
    return np.concatenate([F, G[:, :16], extra, V], 1).astype(np.float32)


def call_outcome_features(X, F, G, action):
    """鳴きの結果(和了率など)を予測するための数値の列。action は (N,) の 0=パス 1=チー 2=ポン 3=大明槓。"""
    import numpy as np
    h = X[:, 0]
    extra = np.stack([h[:, 0:9].sum(1), h[:, 9:18].sum(1), h[:, 18:27].sum(1), h[:, 27:].sum(1)], 1)
    onehot = np.eye(4, dtype=np.float32)[np.asarray(action, dtype=np.int64)]
    return np.concatenate([F, G, extra, onehot], 1).astype(np.float32)


def pf_features(ev, t):
    """押し引きの結果(和了率・放銃率)を予測するための数値の列。
    ev は mjcommon.parse_game の1場面(他家リーチ中)、t はその場面で切る牌の種類。"""
    import numpy as np
    from rules import SAFE_MAX_LEVEL
    x, f = ev["x"], ev["f"]
    hand = x[0].astype(int)
    tiles = [u for u in range(34) if hand[u] > 0]
    lev = {u: int(x[13][u]) - 1 for u in tiles}
    sh = {u: int(x[14][u]) - 1 for u in tiles}
    min_sh = min(sh.values())
    safe = [u for u in tiles if lev[u] <= SAFE_MAX_LEVEL]
    risky = [u for u in tiles if lev[u] > SAFE_MAX_LEVEL]
    best_safe = min((sh[u] for u in safe), default=min_sh + 3) - min_sh
    best_risky = min((sh[u] for u in risky), default=min_sh + 3) - min_sh
    return np.array(list(f.astype(float)) + [
        float(f[1:4].sum()), lev[t], sh[t], sh[t] - min_sh, float(sum(hand[u] for u in safe)),
        best_safe, best_risky, float((hand * x[5]).sum()), float(hand[YAOCHU].sum()), float(ev["melds"]),
        float(hand[0:9].sum()), float(hand[9:18].sum()), float(hand[18:27].sum()), float(hand[27:].sum()),
    ], dtype=np.float32)
