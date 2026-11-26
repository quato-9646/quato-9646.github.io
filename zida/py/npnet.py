"""torch・scikit-learn を使わずに、numpy だけで AI の判断を計算する(ブラウザ内=Pyodide でも動く)。

  NPDiscard … 打牌モデル(DiscardNet と同じ計算)
  NPCall    … 鳴き判断モデル(DecisionNet(4) と同じ計算)
  NPHGB     … 決定木の集まり(scikit-learn の HistGradientBoosting と同じ予測)
学習済みの重みは export_np.py で zida_np.npz に書き出し、load_np() で読む。"""
import json

import numpy as np

NUM_F_SCALE = (18.0, 17.0, 8.0)


def softmax(a):
    a = np.asarray(a, dtype=np.float64)
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))


def _scale_f(f):
    f = np.asarray(f, dtype=np.float32)[None]
    return np.concatenate([f[:, :8], f[:, 8:9] / 18, f[:, 9:10] / 17, f[:, 10:] / 8], axis=1)


def _mlp(h, layers):
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


class NPDiscard:
    """打牌モデル。infer(x, f) → (34個の確率のリスト, 降りの確率)。"""

    def __init__(self, layers, tile_head, fold_head):
        self.layers, self.tile_head, self.fold_head = layers, tile_head, fold_head

    def infer(self, x, f):
        h = np.concatenate([np.asarray(x, dtype=np.float32)[None].reshape(1, -1) / 4, _scale_f(f)], axis=1)
        for w, b in self.layers:
            h = np.maximum(h @ w.T + b, 0)
        scores = h @ self.tile_head[0].T + self.tile_head[1]
        scores = np.where(np.asarray(x)[0][None] == 0, -1e9, scores)
        fold = h @ self.fold_head[0].T + self.fold_head[1]
        return softmax(scores)[0].tolist(), float(sigmoid(fold)[0, 0])


class NPCall:
    """鳴き判断モデル。infer(x, f, g, legal) → 4個の確率(パス・チー・ポン・大明槓)。"""

    def __init__(self, layers, g_scale):
        self.layers, self.g_scale = layers, g_scale

    def infer(self, x, f, g, legal):
        h = np.concatenate([np.asarray(x, dtype=np.float32)[None].reshape(1, -1) / 4, _scale_f(f),
                            np.asarray(g, dtype=np.float32)[None] / self.g_scale], axis=1)
        out = _mlp(h, self.layers)
        out = np.where(np.asarray(legal, dtype=bool)[None], out, -1e9)
        return softmax(out)[0].tolist()


class NPHGB:
    """HistGradientBoosting の予測だけを numpy で再現する。classifier=True なら predict_proba も使える。"""

    def __init__(self, a, classifier):
        self.classifier = classifier
        self.base = float(a["base"])
        self.root = a["root"]            # 木ごとの、根の節の番号(全体の通し番号)
        self.feat, self.thr = a["feat"], a["thr"]
        self.left, self.right = a["left"], a["right"]
        self.leaf = a["leaf"]            # 葉なら1
        self.val = a["val"]
        self.mleft = a["mleft"]
        self.depth = int(a["depth"])

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pos = np.repeat(self.root[:, None], n, axis=1)           # (木, 行)
        cols = np.arange(n)[None, :]
        for _ in range(self.depth + 1):
            isleaf = self.leaf[pos] == 1
            if isleaf.all():
                break
            xv = X[cols, self.feat[pos]]
            go_left = np.where(np.isnan(xv), self.mleft[pos] == 1, xv <= self.thr[pos])
            nxt = np.where(go_left, self.left[pos], self.right[pos])
            pos = np.where(isleaf, pos, nxt)
        return self.base + self.val[pos].sum(axis=0)

    def predict_proba(self, X):
        p = sigmoid(self._raw(X))
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return self._raw(X)


def pack_hgb(m):
    """scikit-learn の HistGradientBoosting を、配列に分解する(export_np.py 用)。"""
    feat, thr, left, right, leaf, val, mleft, root = [], [], [], [], [], [], [], []
    off, depth = 0, 0
    for stage in m._predictors:
        for p in stage:
            nd = p.nodes
            assert not nd["is_categorical"].any(), "カテゴリ特徴は未対応"
            root.append(off)
            feat.append(nd["feature_idx"]); thr.append(nd["num_threshold"])
            left.append(nd["left"].astype(np.int64) + off); right.append(nd["right"].astype(np.int64) + off)
            leaf.append(nd["is_leaf"]); val.append(nd["value"]); mleft.append(nd["missing_go_to_left"])
            depth = max(depth, int(nd["depth"].max()))
            off += len(nd)
    base = float(np.asarray(m._baseline_prediction).ravel()[0])
    return {"base": np.array(base), "root": np.array(root, dtype=np.int64),
            "feat": np.concatenate(feat).astype(np.int64), "thr": np.concatenate(thr).astype(np.float64),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "leaf": np.concatenate(leaf).astype(np.uint8), "val": np.concatenate(val).astype(np.float64),
            "mleft": np.concatenate(mleft).astype(np.uint8), "depth": np.array(depth)}


def load_np(path):
    """zida_np.npz を読み、(打牌モデル, models) を返す。models は review_decisions.load_models と同じ形:
    {"rc": (リーチ, 鳴き, 和了率, 鳴きの結果), "pf": 押し引きの結果}"""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))

    def hgb(name, classifier):
        keys = ("base", "root", "feat", "thr", "left", "right", "leaf", "val", "mleft", "depth")
        return NPHGB({k: z[f"{name}.{k}"] for k in keys}, classifier)

    def layers(prefix, n):
        return [(z[f"{prefix}.w{i}"], z[f"{prefix}.b{i}"]) for i in range(n)]

    disc = NPDiscard(layers("disc", 3), (z["disc.hw"], z["disc.hb"]), (z["disc.fw"], z["disc.fb"]))
    call = NPCall(layers("call", 4), z["call.g_scale"])
    riichi = hgb("riichi", True)
    om = {"win": hgb("om.win", True), "dealin": hgb("om.dealin", True), "tsumo": hgb("om.tsumo", True),
          "loss": {int(k): v for k, v in meta["om.loss"].items()}}
    cm = {"win": hgb("cm.win", True), "dealin": hgb("cm.dealin", True), "pts": hgb("cm.pts", False),
          "tsumo_share": meta["cm.tsumo_share"], "loss": meta["cm.loss"]}
    pf = {"win": hgb("pf.win", True), "dealin": hgb("pf.dealin", True), "pts": hgb("pf.pts", False),
          "tsumo_share": meta["pf.tsumo_share"], "loss": meta["pf.loss"]}
    return disc, {"rc": (riichi, call, om, cm), "pf": pf}
