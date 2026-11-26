"""麻雀AIの共通部品: 牌譜の読み込み・局面の特徴量づくり・モデルの形。

make_dataset.py(学習データ作成)、train.py(学習)、review.py(評価)の3つが
このファイルを読み込んで使う。学習と評価で「同じ特徴量」を作るために、
局面を作る処理はここ1か所にまとめてある。

牌の種類番号: 0〜8=1〜9萬, 9〜17=1〜9筒, 18〜26=1〜9索, 27〜33=東南西北白發中
席の呼び方: 「自分から見た相対席」で 0=自分, 1=下家, 2=対面, 3=上家
"""
import gzip
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:          # ブラウザ内(Pyodide)などtorchが無い環境では、numpy版(npnet.py)で推論する
    torch = nn = None
_Module = nn.Module if nn is not None else object

from rules import SAFE_MAX_LEVEL, discard_shantens, tile_level

DRAW_RE = re.compile(r"([TUVW])(\d+)")      # ツモ: T=席0, U=席1, V=席2, W=席3
DISCARD_RE = re.compile(r"([DEFG])(\d+)")   # 打牌: D=席0, E=席1, F=席2, G=席3
NAMES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("東南西北白發中")
WINDS = "東南西北"
SEAT_NAMES = ["自分", "下家", "対面", "上家"]

# 1つの局面は、牌の表 x (NUM_CH, 34) と、数値の列 f (NUM_F,) の2つ。
# x の各行(34は牌の種類):
#    0      : 自分の手牌(ツモ牌を含む)の枚数
#    1〜4   : 捨て牌(自分・下家・対面・上家)
#    5      : ドラ(表示牌ではなく、ドラになる牌そのもの)
#    6〜9   : 副露で晒している牌(自分・下家・対面・上家)
#    10〜12 : 現物(下家・対面・上家に対して安全な牌。相手の捨て牌と、
#             相手のリーチ後に誰かが捨てた牌)
#    13     : 危険度(誰かがリーチ中のときだけ。手牌の各牌に 安全度の段階+1 を入れる。
#             1=現物 … 7=無筋の中張牌。rules.py の tile_level で計算)
#    14     : 切った後の向聴数(手牌の各牌に 向聴数+1 を入れる。0=手牌にない、1=聴牌になる。
#             rules.py の discard_shantens で計算。切る牌ごとの手の進み具合)
NUM_CH = 15
# f の中身:
#    0〜3   : リーチ中か(自分・下家・対面・上家)。自分は「この打牌がリーチ宣言牌」のとき1
#    4〜7   : 親か(自分・下家・対面・上家)
#    8      : 自分の巡目
#    9      : 牌譜の年(2009年からの経過年数)。昔の対局ほど押す傾向があるため
#    10     : 今の手牌の向聴数+1(どの牌を切っても最良の場合。1=聴牌できる)
NUM_F = 11
BASE_YEAR = 2009
DEFAULT_YEAR = 2023   # 年が分からない牌譜(自分の対局など)に使う値。学習データの範囲内の最近の年


def load(path):
    """mjlogを読み込む(gzip圧縮なら自動で展開)。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return ET.fromstring(data.decode("utf-8"))


def get_names(root):
    """牌譜から4人のプレイヤー名を取り出す。"""
    for el in root:
        if el.tag == "UN" and all(f"n{i}" in el.attrib for i in range(4)):
            return [unquote(el.get(f"n{i}")) for i in range(4)]
    return ["P0", "P1", "P2", "P3"]


def dora_from_indicator(t):
    """ドラ表示牌(種類番号)から、ドラになる牌の種類番号を返す。"""
    if t < 27:                      # 数牌: 9の次は1に戻る
        return t // 9 * 9 + (t % 9 + 1) % 9
    if t < 31:                      # 風牌: 東→南→西→北→東
        return 27 + (t - 27 + 1) % 4
    return 31 + (t - 31 + 1) % 3    # 三元牌: 白→發→中→白


def decode_meld(m):
    """副露コードmを読み、次の4つを返す。
        removed  : 手牌から減る牌(種類番号のリスト)
        shown    : 副露として場に晒される牌(種類番号のリスト)
        new_meld : 新しい副露が1つ増えるか(加槓・北抜きは増えない)
        called   : 他家の捨て牌を鳴いたか(チー・ポン・大明槓)
    鳴いた牌そのものは他家の捨て牌なので、手牌からは減らない。
    注意: 種類は必ず チー → ポン → 加槓 → 北抜き → 槓 の順に判定する。"""
    if m & 0x4:  # チー
        t = m >> 10
        r = t % 3            # 順子のうち、鳴いた牌の位置
        t //= 3
        base = (t // 7) * 9 + t % 7
        seq = [base, base + 1, base + 2]
        removed = [s for i, s in enumerate(seq) if i != r]
        return removed, seq, True, True
    if m & 0x8:  # ポン
        kind = (m >> 9) // 3
        return [kind, kind], [kind] * 3, True, True
    if m & 0x10:  # 加槓(ポン済みの牌に1枚足す。副露の数は増えない)
        kind = (m >> 9) // 3
        return [kind], [kind], False, False
    if m & 0x20:  # 北抜き(三麻のみ)
        return [], [], False, False
    kind = (m >> 8) // 4  # 大明槓 or 暗槓
    if m & 3 == 0:        # 暗槓: 4枚とも手牌から
        return [kind] * 4, [kind] * 4, True, False
    return [kind] * 3, [kind] * 4, True, True  # 大明槓: 3枚が手牌から


AKA_IDS = (16, 52, 88)   # 赤五の牌番号(五萬・五筒・五索)


def meld_detail(m):
    """副露コード m を、画面に描くための形にする。
    戻り値 dict: k=種類(chi/pon/kan/ankan/kakan) / t=牌番号(0〜135)を、置く順に並べたもの
                 s=横向きにする牌の位置(無ければ-1) / a=加槓で足した牌(無ければ-1) / kind=牌の種類番号
    鳴いた相手が 上家(rel=3)なら左端、対面(2)なら真ん中、下家(1)なら右端 の牌を横向きにする。
    加槓・北抜きは、新しい副露ではないので別に扱う(加槓は kind と a だけ入れて k='kakan' を返す)。"""
    rel = m & 3
    if m & 0x4:  # チー: 鳴いた牌を左端に横向きで置く(いつも上家から)
        t = m >> 10
        r = t % 3
        t //= 3
        base = (t // 7) * 9 + t % 7
        seq = [base, base + 1, base + 2]
        ids = [seq[i] * 4 + ((m >> (3 + 2 * i)) & 3) for i in range(3)]
        called = ids[r]
        return {"k": "chi", "t": [called] + [x for i, x in enumerate(ids) if i != r], "s": 0, "a": -1, "kind": -1}
    if m & 0x8:  # ポン
        kind = (m >> 9) // 3
        unused = (m >> 5) & 3
        ids = [kind * 4 + c for c in range(4) if c != unused]
        pos = {3: 0, 2: 1, 1: 2}.get(rel, 1)
        return {"k": "pon", "t": ids, "s": pos, "a": -1, "kind": kind}
    if m & 0x10:  # 加槓: ポンに1枚足す
        kind = (m >> 9) // 3
        return {"k": "kakan", "t": [], "s": -1, "a": kind * 4 + ((m >> 5) & 3), "kind": kind}
    if m & 0x20:  # 北抜き(三麻)
        return None
    kind = (m >> 8) // 4
    ids = [kind * 4 + c for c in range(4)]
    if rel == 0:   # 暗槓: 両端を裏向きに
        return {"k": "ankan", "t": ids, "s": -1, "a": -1, "kind": kind}
    pos = {3: 0, 2: 1, 1: 3}.get(rel, 1)
    return {"k": "kan", "t": ids, "s": pos, "a": -1, "kind": kind}


def build_meld_list(codes):
    """副露コードを順に見て、画面に描く副露のリストを作る(加槓は、元のポンに足す)。"""
    out = []
    for m in codes:
        d = meld_detail(int(m))
        if d is None:
            continue
        if d["k"] == "kakan":
            for g in out:
                if g["k"] == "pon" and g["kind"] == d["kind"]:
                    g["k"], g["a"] = "kakan", d["a"]
                    break
            continue
        out.append(d)
    return out


def parse_game(root, year=DEFAULT_YEAR, trace=False):
    """1つの牌譜を最初から追い、局ごとに「打牌の判断」の一覧を返す。
    trace=True のときは、画面表示用に各場面へ "viz"(4人の河・手牌・赤牌・打牌の通し番号)を足す。

    戻り値: 局のリスト。各局は dict で
        label   : '東1局0本場' のような表示用の文字列
        broken  : 手牌の追跡が合わなくなった局ならTrue(学習には使わない)
        events  : 打牌の判断のリスト。各要素は dict で
            who     : 打った人の席(0〜3)
            junme   : その人の何回目の打牌か
            x, f    : 打牌前の局面(上の説明を参照)
            draw    : 直前のツモ牌(種類番号)。鳴いた直後などはNone
            actual  : 実際に切った牌(種類番号)
            riichi  : このリーチ宣言牌ならTrue
            melds   : その人の副露の数
            fold    : 押し引きの場面の正解。1=安全牌を切った 0=危険牌を切った
                      -1=押し引きの場面ではない(他家のリーチ中でない/安全牌か危険牌の
                      片方しか手牌にない/自分のリーチ宣言牌)
            visible : 場に見えている牌の枚数(34種。自分の手牌・全員の捨て牌・
                      副露・ドラ表示牌の合計。評価ツールの安全度判定に使う)
            ctx     : 点数状況(局の番号・最終局・本場・供託・4人の持ち点・自分の席・親)
    リーチ後の自動の打牌は選択の余地がないので、events に含めない。
    """
    rounds = []
    cur = None
    last_round = 7                      # 最終局の番号(東1=0 … 南4=7)。GOタグの東南の別から決める
    for el in root:
        tag = el.tag
        if tag == "GO":
            last_round = 7 if (int(el.get("type", "0")) & 0x08) else 3
            continue

        if tag == "INIT":  # 局の開始
            seed = [int(v) for v in el.get("seed").split(",")]
            round_idx, honba, sticks = seed[0], seed[1], seed[2]
            scores = [int(v) * 100 for v in el.get("ten").split(",")]
            cur = {"label": f"{WINDS[seed[0] // 4]}{seed[0] % 4 + 1}局{seed[1]}本場",
                   "broken": False, "events": [], "results": [], "init_scores": list(scores)}
            rounds.append(cur)
            hand = np.zeros((4, 34), dtype=np.int8)   # 各自の手牌
            reds = [set() for _ in range(4)]          # 各自が持つ赤牌の牌番号(表示用)
            rivers = [[] for _ in range(4)]           # 各自の河(打牌順。表示用)
            last_discarder, nd, draw_red = None, 0, False
            for i in range(4):
                for t in el.get(f"hai{i}").split(","):
                    hand[i, int(t) // 4] += 1
                    if int(t) in AKA_IDS:
                        reds[i].add(int(t))
            discards = np.zeros((4, 34), dtype=np.int8)    # 各自の捨て牌(鳴かれた牌も含む)
            meld_tiles = np.zeros((4, 34), dtype=np.int8)  # 各自の副露牌
            genbutsu = np.zeros((4, 34), dtype=np.int8)    # genbutsu[k] = 席kに対して安全な牌
            indicators = np.zeros(34, dtype=np.int8)       # ドラ表示牌
            called = np.zeros(34, dtype=np.int8)           # 鳴かれた牌(捨て牌と副露の二重数え防止)
            dora = np.zeros(34, dtype=np.int8)
            dora[dora_from_indicator(seed[5] // 4)] += 1   # seed[5] = 最初のドラ表示牌
            indicators[seed[5] // 4] += 1
            oya = int(el.get("oya"))
            cur["oya"] = oya
            melds = [0] * 4
            meld_codes = [[] for _ in range(4)]       # 各自の副露コード(画面表示用)
            last_draw_tid = [None] * 4                # 直前にツモった牌の牌番号(ツモ切りの判定用)
            junme = [0] * 4
            last_draw = [None] * 4
            last_discard = None
            pending_riichi = [False] * 4
            in_riichi = [False] * 4
            continue
        if cur is None:
            continue

        if tag == "DORA":  # カンドラが増えた
            t = int(el.get("hai")) // 4
            dora[dora_from_indicator(t)] += 1
            indicators[t] += 1
        elif tag == "N":   # 副露
            who, m = int(el.get("who")), int(el.get("m"))
            meld_codes[who].append(m)
            last_draw_tid[who] = None
            removed, shown, new_meld, was_called = decode_meld(m)
            for t in removed:
                hand[who, t] -= 1
            for t in shown:
                meld_tiles[who, t] += 1
            if new_meld:
                melds[who] += 1
            if was_called and last_discard is not None:
                called[last_discard] += 1
                if last_discarder is not None and rivers[last_discarder]:
                    rivers[last_discarder][-1][3] = 1      # 河の最後の牌が鳴かれた
            for rid in list(reds[who]):                    # 副露に使った赤牌は手牌から外す
                if hand[who, rid // 4] <= 0:
                    reds[who].discard(rid)
            last_draw[who] = None   # 鳴いた直後はツモ牌がない
            if hand[who].min() < 0:
                cur["broken"] = True
        elif tag == "REACH":
            if el.get("step") == "1":
                pending_riichi[int(el.get("who"))] = True
            elif el.get("ten"):
                scores = [int(v) * 100 for v in el.get("ten").split(",")]
                sticks += 1
        elif tag in ("AGARI", "RYUUKYOKU"):     # 局の結果。あとで各場面に紐づける
            sc = el.get("sc")
            delta = [int(v) * 100 for v in sc.split(",")[1::2]] if sc else [0, 0, 0, 0]
            cur["results"].append({"tag": tag, "who": int(el.get("who", -1)),
                                   "from": int(el.get("fromWho", -1)), "delta": delta,
                                   "attrs": dict(el.attrib) if trace else None})
        else:
            m = DRAW_RE.fullmatch(tag)
            if m:
                who = "TUVW".index(m.group(1))
                t = int(m.group(2)) // 4
                hand[who, t] += 1
                last_draw[who] = t
                last_draw_tid[who] = int(m.group(2))
                if int(m.group(2)) in AKA_IDS:
                    reds[who].add(int(m.group(2)))
                draw_red = int(m.group(2)) in AKA_IDS
                continue
            m = DISCARD_RE.fullmatch(tag)
            if m:
                who = "DEFG".index(m.group(1))
                tid = int(m.group(2))
                t = tid // 4
                junme[who] += 1
                if not in_riichi[who]:
                    # 検算: 打牌時の手牌は 14 - 3×副露数 枚のはず
                    if hand[who, t] <= 0 or hand[who].sum() != 14 - 3 * melds[who]:
                        cur["broken"] = True
                    x = np.empty((NUM_CH, 34), dtype=np.uint8)
                    x[0] = hand[who]
                    for k in range(4):            # 自分から見て 自分→下家→対面→上家
                        x[1 + k] = discards[(who + k) % 4]
                        x[6 + k] = meld_tiles[(who + k) % 4]
                    x[5] = dora
                    for k in range(1, 4):
                        x[9 + k] = genbutsu[(who + k) % 4]
                    f = np.zeros(NUM_F, dtype=np.uint8)
                    f[0] = pending_riichi[who]
                    for k in range(1, 4):
                        f[k] = in_riichi[(who + k) % 4]
                    for k in range(4):
                        f[4 + k] = (who + k) % 4 == oya
                    f[8] = min(junme[who], 255)
                    f[9] = min(max(year - BASE_YEAR, 0), 255)
                    x[14] = 0
                    sh = discard_shantens(x[0], melds[who])   # 各牌を切った後の向聴数
                    for u, v in sh.items():
                        x[14][u] = v + 1
                    f[10] = min(sh.values()) + 1 if sh else 0
                    visible = np.clip(hand[who].astype(np.int16) + discards.sum(0) - called
                                      + meld_tiles.sum(0) + indicators, 0, 255).astype(np.uint8)

                    # 押し引き: 誰かがリーチ中なら、手牌の各牌の危険度を計算する
                    x[13] = 0
                    fold = -1
                    seats = [k for k in range(1, 4) if f[k]]
                    if seats:
                        level = {u: tile_level(u, x, seats, visible)
                                 for u in range(34) if x[0][u] > 0}
                        for u, lv in level.items():
                            x[13][u] = lv + 1
                        has_safe = any(lv <= SAFE_MAX_LEVEL for lv in level.values())
                        has_risky = any(lv > SAFE_MAX_LEVEL for lv in level.values())
                        if has_safe and has_risky and not pending_riichi[who] and t in level:
                            fold = int(level[t] <= SAFE_MAX_LEVEL)

                    cur["events"].append({
                        "who": who, "junme": junme[who], "x": x, "f": f,
                        "draw": last_draw[who], "actual": t,
                        "riichi": pending_riichi[who], "melds": melds[who],
                        "visible": visible, "fold": fold,
                        "ctx": {"round_idx": round_idx, "last_round": last_round, "honba": honba,
                                "sticks": sticks, "scores": list(scores), "me": who, "oya": oya,
                                "seat": (who - oya) % 4},
                    })
                    if trace:
                        cur["events"][-1]["viz"] = {
                            "t": nd, "hands": hand.copy(), "melds": meld_tiles.copy(),
                            "rivers": [[list(e) for e in rv] for rv in rivers],
                            "meld_list": [build_meld_list(meld_codes[k]) for k in range(4)],
                            "reds": [sorted({rid // 4 for rid in rs}) for rs in reds],
                            "draw_red": bool(draw_red and last_draw[who] is not None),
                            "actual_red": tid in AKA_IDS, "riichi_seats": [bool(v) for v in in_riichi]}
                hand[who, t] -= 1
                discards[who, t] += 1
                last_discard = t
                last_discarder = who
                rivers[who].append([t, int(tid in AKA_IDS), int(pending_riichi[who]), 0, int(last_draw_tid[who] == tid)])
                last_draw_tid[who] = None
                reds[who].discard(tid)
                nd += 1
                genbutsu[who, t] = 1              # 自分の捨て牌は自分に対して安全(現物)
                for k in range(4):                # 相手がリーチ中なら、その相手に対して安全になる
                    if k != who and in_riichi[k]:
                        genbutsu[k, t] = 1
                if hand[who].min() < 0:
                    cur["broken"] = True
                if pending_riichi[who]:  # リーチ宣言牌の次の打牌から自動になる
                    in_riichi[who] = True
                    pending_riichi[who] = False
    for r in rounds:
        res = r["results"]
        for ev in r["events"]:                 # その人のこの局の結果
            w = ev["who"]
            wins = [a for a in res if a["tag"] == "AGARI" and a["who"] == w]
            ev["win"] = int(bool(wins))
            ev["tsumo"] = int(any(a["from"] == w for a in wins))
            ev["dealin"] = int(any(a["tag"] == "AGARI" and a["from"] == w and a["who"] != w for a in res))
            ev["delta"] = int(sum(a["delta"][w] for a in res))
    return rounds


class DiscardNet(_Module):
    """局面 (x: 15×34, f: 11個) から、次の2つを出す全結合ネットワーク。
        scores     : 各牌を切る点数(34個)
        fold_logit : 押し引きの点数。大きいほど「プロなら安全牌を切る」(降り)
    2つは同じ土台(中間層)を共有して、いっしょに学習する。"""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(NUM_CH * 34 + NUM_F, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),   # Dropout: 学習中だけ一部を無効にして過学習を防ぐ
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.tile_head = nn.Linear(256, 34)   # どの牌を切るか
        self.fold_head = nn.Linear(256, 1)    # 降りるか(安全牌を切るか)

    def forward(self, x, f):
        # x, f は uint8 の整数。小数にして 0〜1 前後に縮め、1列に並べてから入れる
        f = f.float()
        f = torch.cat([f[:, :8], f[:, 8:9] / 18, f[:, 9:10] / 17, f[:, 10:] / 8], dim=1)  # 巡目・年・向聴数は割って縮める
        h = self.trunk(torch.cat([x.float().flatten(1) / 4, f], dim=1))
        scores = self.tile_head(h)
        # 手牌に無い牌は切れないので、点数を極端に低くして選ばれないようにする
        scores = scores.masked_fill(x[:, 0] == 0, -1e9)
        return scores, self.fold_head(h).squeeze(1)
