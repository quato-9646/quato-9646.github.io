/* ブラウザの中でPython(Pyodide)を動かして、牌譜を解析する。画面が固まらないように、別スレッド(Worker)で動かす。 */
const post = m => self.postMessage(m);
let py = null, initP = null;

async function init() {
  const {loadPyodide} = await import("./pyodide/pyodide.mjs");
  post({t: "status", m: "AIエンジンを読み込み中…(初回だけ少し時間がかかります)", p: 8});
  py = await loadPyodide({indexURL: new URL("pyodide/", self.location.href).href});
  post({t: "status", m: "計算ライブラリを読み込み中…", p: 40});
  await py.loadPackage("numpy");
  post({t: "status", m: "AIのモデルを読み込み中…", p: 65});
  const list = await (await fetch("py/files.json")).json();
  py.FS.mkdirTree("/app/assets");
  for (const f of list) {
    const r = await fetch("py/" + f);
    if (!r.ok) throw new Error(f + " を読み込めませんでした");
    py.FS.writeFile("/app/" + f, new Uint8Array(await r.arrayBuffer()));
  }
  py.runPython("import sys\nsys.path.insert(0, '/app')\nimport browser_entry\nbrowser_entry.load_model()");
  post({t: "ready"});
}

function clean(err) {
  const s = String((err && err.message) || err);
  const m = s.match(/LogError: (.*)/);
  if (m) return {user: true, msg: m[1].trim()};
  const last = s.trim().split("\n").pop();
  return {user: false, msg: last};
}

initP = init().catch(err => { post({t: "initerror", msg: String((err && err.message) || err)}); throw err; });
initP.catch(() => {});

self.onmessage = async e => {
  const m = e.data;
  try {
    await initP;
    if (m.t === "names") {
      py.globals.set("xml_in", m.xml);
      const r = py.runPython("import json\njson.dumps(browser_entry.player_names(xml_in))");
      post({t: "names", id: m.id, names: JSON.parse(r)});
    } else if (m.t === "review") {
      py.globals.set("xml_in", m.xml);
      py.globals.set("seat_in", m.seat);
      py.globals.set("title_in", m.title);
      py.globals.set("year_in", m.year);
      py.globals.set("prog", (a, b) => post({t: "prog", a, b}));
      const html = py.runPython("browser_entry.review(xml_in, seat_in, title_in, year_in, prog)");
      post({t: "done", id: m.id, html});
    }
  } catch (err) {
    const c = clean(err);
    post({t: "error", id: m.id, user: c.user, msg: c.msg});
  }
};
