"""天鳳の「tenhou.net/6」形式のJSON(雀魂の牌譜を変換するツールなどが出力する形式)を、
天鳳のmjlog(XML)に変換する。変換後は、天鳳の牌譜と同じ解析の流れに乗せる。
対応: 4人麻雀・半荘。三麻・東風戦は入口で断る。"""
import json
import re
from urllib.parse import quote

from make_viewer import YAKU

AKA = {51: 4, 52: 13, 53: 22}
FLOW = {"九種九牌": "yao9", "四風連打": "kaze4", "四家立直": "reach4", "三家和了": "ron3",
        "四槓散了": "kan4", "流し満貫": "nm"}
LIMITS = [("三倍満", 4), ("倍満", 3), ("跳満", 2), ("満貫", 1), ("役満", 5)]


class T6Error(Exception):
    pass


def kind_of(n):
    n = int(n)
    if n in AKA:
        return AKA[n], True
    if 11 <= n <= 19:
        return n - 11, False
    if 21 <= n <= 29:
        return 9 + n - 21, False
    if 31 <= n <= 39:
        return 18 + n - 31, False
    if 41 <= n <= 47:
        return 27 + n - 41, False
    raise T6Error(f"牌の番号が読めません: {n}")


class Round:
    """1局ぶんの状態(牌の番号の割り当てと、手牌・河・副露)。"""

    def __init__(self):
        self.used = [set() for _ in range(34)]
        self.hand = [[] for _ in range(4)]
        self.last_discard = [None] * 4
        self.melds = [[] for _ in range(4)]

    def alloc(self, n):
        k, red = kind_of(n)
        if red:
            tid = k * 4
            self.used[k].add(0)
            return tid
        for c in (1, 2, 3, 0):
            if c not in self.used[k] and not (c == 0 and k in (4, 13, 22)):
                self.used[k].add(c)
                return k * 4 + c
        for c in (0, 1, 2, 3):
            if c not in self.used[k]:
                self.used[k].add(c)
                return k * 4 + c
        return k * 4 + 1

    def take(self, p, n):
        """手牌から n に当たる牌を1枚抜いて、その牌番号を返す(無ければ新しく作る)。"""
        k, red = kind_of(n)
        for i, tid in enumerate(self.hand[p]):
            if tid // 4 == k and ((tid == k * 4) == red if k in (4, 13, 22) else True):
                return self.hand[p].pop(i)
        return self.alloc(n)


def _split(s):
    """'c141516' / '2525p25' などを (文字, 文字の前の牌の数, 牌のリスト) にする。"""
    letter = re.search(r"[cpmka]", s).group(0)
    pos = s.index(letter)
    body = s.replace(letter, "")
    nums = [int(body[i:i + 2]) for i in range(0, len(body), 2)]
    return letter, pos // 2, nums


def _copy(tid):
    return tid % 4


def meld_code(letter, before, nums, ids, rel):
    """鳴きの文字列から天鳳のmeldコード(整数)を作る。idsは各牌の牌番号(numsと同じ順)。"""
    if letter == "c":
        order = sorted(range(3), key=lambda i: ids[i] // 4)
        seq = [ids[i] for i in order]
        r = order.index(0)                     # 先頭の牌が鳴いた牌
        base = seq[0] // 4
        t = (base // 9) * 7 + base % 9
        m = ((t * 3 + r) << 10) | 0x4 | 3
        for i, tid in enumerate(seq):
            m |= _copy(tid) << (3 + 2 * i)
        return m
    kind = ids[0] // 4
    if letter == "p":
        used = {_copy(i) for i in ids}
        unused = next(c for c in range(4) if c not in used)
        return ((kind * 3 + 0) << 9) | (unused << 5) | 0x8 | rel
    if letter == "m":
        return ((kind * 4 + _copy(ids[0])) << 8) | rel
    if letter == "a":
        return ((kind * 4 + _copy(ids[0])) << 8)
    if letter == "k":
        added = ids[0]
        return ((kind * 3 + 0) << 9) | (_copy(added) << 5) | 0x10 | rel
    raise T6Error("鳴きの形が読めません")


def _rel(before, n_tiles):
    """文字の前にある牌の数から、鳴いた相手(1=下家 2=対面 3=上家)を決める。"""
    if before == 0:
        return 3
    if before == 1:
        return 2
    return 1


def _src(q, rel):
    return {3: (q - 1) % 4, 2: (q + 2) % 4, 1: (q + 1) % 4}[rel]


def convert(text):
    data = json.loads(text)
    if isinstance(data, list):
        data = {"log": data}
    log = data.get("log")
    if not log:
        raise T6Error("対局の記録(log)が見つかりません。")
    names = data.get("name") or ["P0", "P1", "P2", "P3"]
    if len(names) != 4:
        raise T6Error("三人麻雀には対応していません。現在は4人麻雀の半荘戦のみです。")
    out = ['<mjloggm ver="2.3">', '<SHUFFLE seed="" ref=""/>', '<GO type="9" lobby="0"/>',
           "<UN " + " ".join(f'n{i}="{quote(str(names[i]))}"' for i in range(4)) + ' dan="0,0,0,0" rate="1500,1500,1500,1500" sx="M,M,M,M"/>',
           '<TAIKYOKU oya="0"/>']
    final_scores = None
    last_round_idx = len(log) - 1
    for ri, rd in enumerate(log):
        final_scores = _round(rd, out, ri == last_round_idx, data)
    out.append("</mjloggm>")
    return "".join(out)


def _round(rd, out, is_last, data):
    head = rd[0]
    kyoku, honba, kyotaku = int(head[0]), int(head[1]), int(head[2])
    scores = [int(v) for v in rd[1]]
    dora_ind = [int(v) for v in rd[2]]
    ura_ind = [int(v) for v in rd[3]] if len(rd) > 3 and rd[3] else []
    players = [(rd[4 + 3 * p], rd[5 + 3 * p], rd[6 + 3 * p]) for p in range(4)]
    result = rd[16] if len(rd) > 16 else None
    R = Round()
    oya = kyoku % 4
    for p in range(4):
        R.hand[p] = [R.alloc(n) for n in players[p][0]]
    dora_ids = [R.alloc(n) for n in dora_ind]
    ura_ids = [R.alloc(n) for n in ura_ind]
    ev = []
    ev.append(f'<INIT seed="{kyoku},{honba},{kyotaku},0,0,{dora_ids[0]}" '
              f'ten="{",".join(str(s // 100) for s in scores)}" oya="{oya}" ' +
              " ".join(f'hai{p}="{",".join(str(i) for i in R.hand[p])}"' for p in range(4)) + "/>")
    cur = list(scores)
    riichi_n = 0
    kans = 0
    idx = [0] * 4
    draws = [players[p][1] for p in range(4)]
    disc = [players[p][2] for p in range(4)]
    last_tile = {"id": None, "from": None}
    p = oya
    riichi_last = None
    win_tile = {}
    steps = 0
    while steps < 400:
        steps += 1
        if idx[p] >= len(draws[p]):
            break
        d = draws[p][idx[p]]
        called_from = None
        if isinstance(d, str):
            letter, before, nums = _split(d)
            rel = _rel(before, len(nums))
            src = _src(p, 3 if letter == "c" else rel)
            called_id = R.last_discard[src]
            # 鳴いた牌 = 文字の直後の牌(チーは先頭)
            ci = 0 if letter == "c" else before
            ids = [None] * len(nums)
            ids[ci] = called_id if called_id is not None else R.alloc(nums[ci])
            for i, n in enumerate(nums):
                if i != ci:
                    ids[i] = R.take(p, n)
            code = meld_code(letter, before, [nums[ci]] + [n for i, n in enumerate(nums) if i != ci],
                             [ids[ci]] + [t for i, t in enumerate(ids) if i != ci], 3 if letter == "c" else rel)
            R.melds[p].append(code)
            ev.append(f'<N who="{p}" m="{code}"/>')
            if letter == "m":
                kans += 1
                if kans < len(dora_ids):
                    ev.append(f'<DORA hai="{dora_ids[kans]}"/>')
        else:
            tid = R.alloc(int(d))
            R.hand[p].append(tid)
            ev.append(f'<{"TUVW"[p]}{tid}/>')
            win_tile[p] = tid
        if idx[p] >= len(disc[p]):
            break
        e = disc[p][idx[p]]
        idx[p] += 1
        stay = False
        riichi = False
        if isinstance(e, str):
            if e[0] == "r":
                riichi = True
                e = int(e[1:])
            else:
                letter, before, nums = _split(e)
                if letter == "k":
                    ci = before
                    added = R.take(p, nums[ci])
                    ids = [added] + [R.alloc(n) for i, n in enumerate(nums) if i != ci]
                    code = ((ids[0] // 4 * 3) << 9) | (_copy(added) << 5) | 0x10 | 1
                    R.melds[p].append(code)
                    ev.append(f'<N who="{p}" m="{code}"/>')
                else:  # a: 暗槓
                    ids = [R.take(p, n) for n in nums]
                    code = ((ids[0] // 4 * 4 + _copy(ids[0])) << 8)
                    R.melds[p].append(code)
                    ev.append(f'<N who="{p}" m="{code}"/>')
                kans += 1
                if kans < len(dora_ids):
                    ev.append(f'<DORA hai="{dora_ids[kans]}"/>')
                stay = True
        if not stay:
            if e == 0:                         # 大明槓のあとの空の打牌
                stay = True
            else:
                tsumo = (int(e) == 60)
                if tsumo:
                    tid = R.hand[p].pop() if R.hand[p] else R.alloc(11)
                else:
                    tid = R.take(p, int(e))
                if riichi:
                    ev.append(f'<REACH who="{p}" step="1"/>')
                ev.append(f'<{"DEFG"[p]}{tid}/>')
                R.last_discard[p] = tid
                last_tile = {"id": tid, "from": p}
                if riichi:
                    cur[p] -= 1000
                    riichi_n += 1
                    riichi_last = len(ev)
                    ev.append(f'<REACH who="{p}" step="2" ten="{",".join(str(s // 100) for s in cur)}"/>')
        if stay:
            continue
        nxt = (p + 1) % 4
        for q in range(1, 4):
            c = (p + q) % 4
            if idx[c] < len(draws[c]) and isinstance(draws[c][idx[c]], str):
                lt, bf, nm = _split(draws[c][idx[c]])
                if lt in "cpm" and _src(c, 3 if lt == "c" else _rel(bf, len(nm))) == p:
                    nxt = c
                    break
        p = nxt
    # --- 結果 ---
    sticks = kyotaku + riichi_n
    if result:
        kind = result[0]
        if kind == "和了":
            first = True
            for i in range(1, len(result), 2):
                delta = [int(v) for v in result[i]]
                det = result[i + 1]
                who, frm = int(det[0]), int(det[1])
                text = det[3]
                yk = det[4:]
                m = re.search(r"(\d+)符(\d+)飜", text)
                fu = int(m.group(1)) if m else 0
                lim = 0
                for nm_, v in LIMITS:
                    if nm_ in text:
                        lim = v
                        break
                gain = delta[who] - 300 * honba - (1000 * sticks if first else 0)
                if gain <= 0:
                    gain = abs(delta[who])
                ys, yms = [], []
                for y in yk:
                    mm = re.match(r"(.+?)\((.+?)\)", y)
                    if not mm:
                        continue
                    name, han = mm.group(1), mm.group(2)
                    if name not in YAKU:
                        continue
                    yid = YAKU.index(name)
                    if "役満" in han:
                        yms.append(yid)
                    else:
                        hn = int(re.sub(r"\D", "", han) or 0)
                        ys += [yid, hn]
                hand = list(R.hand[who])
                if frm == who:
                    machi = win_tile.get(who, hand[-1] if hand else 0)
                else:
                    machi = last_tile["id"] if last_tile["id"] is not None else 0
                    hand.append(machi)
                nd = 1 + kans
                at = [f'ba="{honba},{sticks if first else 0}"', f'hai="{",".join(str(i) for i in sorted(hand))}"']
                if R.melds[who]:
                    at.append(f'm="{",".join(str(c) for c in R.melds[who])}"')
                at += [f'machi="{machi}"', f'ten="{fu},{gain},{lim}"']
                if ys:
                    at.append(f'yaku="{",".join(str(v) for v in ys)}"')
                if yms:
                    at.append(f'yakuman="{",".join(str(v) for v in yms)}"')
                at.append(f'doraHai="{",".join(str(i) for i in dora_ids[:nd])}"')
                if any(y.startswith("立直") or y.startswith("両立直") for y in yk) and ura_ids:
                    at.append(f'doraHaiUra="{",".join(str(i) for i in ura_ids[:nd])}"')
                at += [f'who="{who}"', f'fromWho="{frm}"',
                       'sc="' + ",".join(f"{cur[k] // 100},{delta[k] // 100}" for k in range(4)) + '"']
                if is_last:
                    at.append(_owari(scores, delta, data))
                ev.append("<AGARI " + " ".join(at) + "/>")
                first = False
        else:
            delta = [int(v) for v in result[1]] if len(result) > 1 and result[1] else [0, 0, 0, 0]
            at = [f'ba="{honba},{sticks}"',
                  'sc="' + ",".join(f"{cur[k] // 100},{delta[k] // 100}" for k in range(4)) + '"']
            if kind in FLOW:
                at.append(f'type="{FLOW[kind]}"')
            else:
                tenpai = [True] * 4 if kind == "全員聴牌" else [d > 0 for d in delta] if any(delta) else [False] * 4
                for k in range(4):
                    if tenpai[k]:
                        at.append(f'hai{k}="{",".join(str(i) for i in sorted(R.hand[k]))}"')
            if is_last:
                at.append(_owari(scores, delta, data))
            ev.append("<RYUUKYOKU " + " ".join(at) + "/>")
    out.extend(ev)
    return [s + d for s, d in zip(scores, [0, 0, 0, 0])]


def _owari(scores, delta, data):
    fin = [(s + d) // 100 for s, d in zip(scores, delta)]
    pts = data.get("final_points") or None
    parts = []
    for i in range(4):
        parts.append(str(fin[i]))
        parts.append(f"{(fin[i] * 100 - 25000) / 1000:.1f}")
    return 'owari="' + ",".join(parts) + '"'
