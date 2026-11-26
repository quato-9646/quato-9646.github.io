"""ブラウザ内(Pyodide)から呼ぶ入口。牌譜(XML文字列)を受け取り、レビュー画面のHTMLを返す。
サーバーは使わず、利用者のブラウザの中で全部計算する。"""
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import make_viewer
import npnet

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CHARS = 6_000_000
_MODEL = None


class LogError(Exception):
    pass


def _validate(xml):
    if len(xml) > MAX_CHARS:
        raise LogError("牌譜が大きすぎます。")
    try:
        root = ET.fromstring(xml)
    except Exception:
        raise LogError("牌譜(XML)として読めませんでした。天鳳の牌譜の中身を、そのまま貼ってください。")
    if root.tag != "mjloggm":
        raise LogError("天鳳の牌譜ではないようです。")
    go = root.find("GO")
    if go is None:
        raise LogError("牌譜の形式が読み取れませんでした。")
    t = int(go.get("type", "0"))
    if t & 0x10:
        raise LogError("三人麻雀には対応していません。現在は4人麻雀の半荘戦のみです。")
    if not t & 0x08:
        raise LogError("東風戦には対応していません。現在は4人麻雀の半荘戦のみです。")
    if not any(el.tag == "INIT" for el in root):
        raise LogError("対局の内容が入っていません。")
    last = [el for el in root if el.tag in ("AGARI", "RYUUKYOKU")]
    if not last or last[-1].get("owari") is None:
        raise LogError("最後まで終わっていない対局です。半荘が終了した牌譜を使ってください。")
    return root


def player_names(xml):
    """4人の名前を返す(入力欄の席の選択肢用)。読めなければエラーの文字列を投げる。"""
    root = _validate(xml)
    for el in root:
        if el.tag == "UN" and all(f"n{i}" in el.attrib for i in range(4)):
            return [unquote(el.get(f"n{i}")) for i in range(4)]
    return ["P0", "P1", "P2", "P3"]


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = npnet.load_np(os.path.join(HERE, "zida_np.npz"))
    return _MODEL


def review(xml, seat, title, year_hint=None, progress=None):
    """seat(0〜3)の視点でレビューを作り、完成したHTML(文字列)を返す。"""
    _validate(xml)
    model, models = load_model()
    m = re.match(r"(\d{4})", year_hint or "")
    name = f"{m.group(1) if m else 2024}0101gm-0000-0000-00000000.mjlog"   # 牌譜の年(ルールの判定に使う)を、ファイル名で渡す
    path = os.path.join("/tmp", name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    g = make_viewer.build_game(model, models, path, str(seat), progress)
    g["title"] = title or "アップロードした牌譜"
    return make_viewer.make_html([g])
