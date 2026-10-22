import hashlib
import html
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).parent

# ==========================================================
# Konfigurasi halaman
# ==========================================================
st.set_page_config(
    page_title="Horses or Humans AI",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = ["horses", "humans"]  # urutan sesuai folder dataset saat training
EMOJI = {"horses": "🐴", "humans": "🧍"}
SAMPLE_DIR = BASE_DIR / "sample_image"
IMG_TYPES = {".jpg", ".jpeg", ".png"}
MODES = ["YOLOv8", "CNN Keras", "Bandingkan"]
SOURCES = ["Unggah", "Kamera", "Contoh"]


def load_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        extra = "".join(f".c{i}{{--i:{i}}}" for i in range(101))  # nilai untuk animasi hitung naik
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}{extra}</style>", unsafe_allow_html=True)


load_css()


# ==========================================================
# Pemuatan model (lazy). PyTorch/Ultralytics selalu SEBELUM TensorFlow.
# ==========================================================
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_yolo():
    from ultralytics import YOLO

    return YOLO(str(BASE_DIR / "model" / "best.pt"))


@st.cache_resource(show_spinner="Memuat model CNN Keras...")
def load_keras():
    import tensorflow as tf

    return tf.keras.models.load_model(str(BASE_DIR / "model" / "classifier_model.h5"), compile=False)


def predict_keras(img: Image.Image) -> dict:
    model = load_keras()
    arr = np.array(img.resize((224, 224)), dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    t0 = time.perf_counter()
    probs = model.predict(arr, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    return {"name": "CNN Keras", "labels": CLASS_NAMES, "probs": np.asarray(probs), "ms": ms}


def predict_yolo(img: Image.Image) -> dict:
    model = load_yolo()
    t0 = time.perf_counter()
    r = model(img, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    if model.task != "classify":
        st.error("best.pt adalah model deteksi, bukan klasifikasi. Gunakan model YOLOv8-cls.")
        st.stop()
    labels = [r.names[i] for i in range(len(r.names))]
    probs = r.probs.data.cpu().numpy()
    return {"name": "YOLOv8", "labels": labels, "probs": probs, "ms": ms}


# ==========================================================
# Pembuat HTML (kartu, grafik, treemap, radar)
# ==========================================================
ARROW = (
    '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 10.5V1.5M2 5.5l4-4 4 4" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
ICON = (
    '<svg class="tm-icon" viewBox="0 0 14 14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="4" '
    'fill="currentColor" opacity=".9"/><path class="sp" d="M7 3.6l1 2.4 2.4 1-2.4 1L7 10.4 6 8 3.6 7 6 6z"/></svg>'
)


def squash(s: str) -> str:
    """Jadikan HTML satu baris agar Markdown tidak salah menafsirkan indentasi."""
    return "".join(line.strip() for line in s.splitlines())


def class_of(label: str) -> str:
    return label if label in EMOJI else "other"


def delta_html(diff: float, text: str) -> str:
    up = diff >= 0
    return f'<span class="ab-delta {"up" if up else "down"}">{ARROW}{text}</span>'


def count_up(value: float, suffix: str = "") -> str:
    """Angka besar; browser modern menghitung naik dari 0 lewat CSS."""
    whole, dec = f"{value:.1f}".split(".")
    w = int(whole)
    if 0 <= w <= 100:
        return f'<span class="cu c{w}"><span class="real">{whole}</span></span><small>.{dec}{suffix}</small>'
    return f"{whole}<small>.{dec}{suffix}</small>"


# ---------- kartu hasil prediksi ----------
def result_html(res: dict, threshold: float) -> tuple[str, str, float]:
    probs, labels = res["probs"], res["labels"]
    idx = int(np.argmax(probs))
    label, conf = str(labels[idx]), float(probs[idx])
    diff = (conf - threshold) * 100

    bars = ""
    for i, name in enumerate(labels):
        pct = float(probs[i]) * 100
        cls = "" if i == idx else "rendah"
        bars += (
            f'<div class="bar-row"><div class="bar-head"><span>{EMOJI.get(name, "")} {html.escape(str(name))}</span>'
            f"<b>{pct:.1f}%</b></div>"
            f'<div class="bar-bg"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div></div>'
        )
    warn = ""
    if diff < 0:
        warn = f'<div class="ab-warn">Keyakinan di bawah ambang {threshold * 100:.0f}%. Hasil kurang bisa diandalkan.</div>'

    out = squash(
        f"""
        <div class="ab-label">{html.escape(res['name'].upper())} · keyakinan</div>
        <div class="ab-big">{count_up(conf * 100, '%')}{delta_html(diff, f'{abs(diff):.1f} poin')}</div>
        <div class="ab-note">dibanding ambang {threshold * 100:.0f}%</div>
        <div class="tile {class_of(label)}" style="margin-top:1rem">
            <span class="tn">Prediksi</span>
            <span class="te">{EMOJI.get(label, '❓')}</span>
            <span class="tn big">{html.escape(label)}</span>
        </div>
        <div class="bars">{bars}</div>
        <div class="ab-foot"><span>Waktu inferensi</span><b>{res['ms']:.0f} ms</b></div>
        {warn}
        """
    )
    return out, label, conf


# ---------- grafik ----------
def smooth_path(P):
    """Garis halus melewati titik-titik (catmull-rom -> bezier)."""
    n = len(P)
    g = lambda i: P[max(0, min(n - 1, i))]
    d = f"M{P[0][0]:.1f},{P[0][1]:.1f}"
    for i in range(n - 1):
        p0, p1, p2, p3, k = g(i - 1), g(i), g(i + 1), g(i + 2), 0.18
        d += (
            f"C{p1[0] + (p2[0] - p0[0]) * k:.1f},{p1[1] + (p2[1] - p0[1]) * k:.1f} "
            f"{p2[0] - (p3[0] - p1[0]) * k:.1f},{p2[1] - (p3[1] - p1[1]) * k:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        )
    return d


def placeholder(icon: str, title: str, text: str) -> str:
    return f'<div class="kosong"><div class="ikon">{icon}</div><b>{title}</b>{text}</div>'


def spark_inset(rows: list, uid: str) -> str:
    """Grafik kecil di dalam kartu (gaya 'Last 30 days')."""
    vals = [r["Keyakinan (%)"] for r in rows][-30:]
    if len(vals) < 2:
        return '<div class="inset spark"><div class="cap">30 prediksi terakhir</div><div class="inset-empty">Butuh 2 prediksi</div></div>'
    W, H, top, bot = 220, 130, 34, 12
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 4.0)
    mn, mx = lo - span * 0.45, hi + span * 0.12
    Y = lambda v: top + (1 - (v - mn) / (mx - mn)) * (H - top - bot)
    X = lambda i: 10 + i / (len(vals) - 1) * (W - 62)
    P = [(X(i), Y(v)) for i, v in enumerate(vals)]
    d = smooth_path(P)
    grid = "".join(
        f'<line class="grid" x1="0" x2="{W}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
        f'<text class="tick" x="{W - 8}" y="{Y(t) - 4:.1f}" text-anchor="end">{t:g}%</text>'
        for t in sorted({round(lo, 1), round((lo + hi) / 2, 1), round(hi, 1)})
    )
    lx, ly = P[-1]
    return squash(
        f"""
        <div class="inset spark"><div class="cap">30 prediksi terakhir</div>
        <svg viewBox="0 0 {W} {H}" role="img" aria-label="Tren keyakinan">
        <defs><linearGradient id="l{uid}" x1="0" x2="1"><stop offset="0" stop-color="#22D3EE"/><stop offset="1" stop-color="#3BF08C"/></linearGradient>
        <linearGradient id="a{uid}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#3BF08C" stop-opacity=".22"/><stop offset="1" stop-color="#3BF08C" stop-opacity="0"/></linearGradient></defs>
        {grid}
        <path class="area" style="fill:url(#a{uid})" d="{d}L{lx:.1f},{H}L{P[0][0]:.1f},{H}Z"/>
        <path class="line" style="stroke:url(#l{uid})" pathLength="1" d="{d}"/>
        <circle class="dot" r="4.5" cx="{lx:.1f}" cy="{ly:.1f}"/>
        </svg></div>
        """
    )


def waves_html(rows: list) -> str:
    """Gelombang cermin: peluang kelas pertama di atas, kelas kedua di bawah."""
    models: dict[str, list] = {}
    for r in reversed(rows):  # dari yang terlama
        models.setdefault(r["Model"], []).append(r)
    series = [m for m in models.values() if len(m) >= 2][:2]
    if not series:
        return '<div class="inset wave"><div class="inset-empty">Lakukan minimal 2 prediksi dengan model yang sama untuk melihat gelombang probabilitas.</div></div>'
    W, H, pad = 520, 170, 16
    mid, amp = H / 2, H / 2 - 14
    paths = ""
    for k, ms in enumerate(series):
        ms = ms[-30:]
        n = len(ms)
        X = lambda i: pad + 6 + i / (n - 1) * (W - 2 * pad - 12)
        for g, idx, sign in (("d", 0, -1), ("m", 1, 1)):
            P = [(X(i), mid + sign * (3 + float(r["_p"][idx] if len(r["_p"]) > idx else 0) * amp)) for i, r in enumerate(ms)]
            op = 1 if k == 0 else 0.5
            sw = 2.5 if k == 0 else 1.6
            paths += f'<path class="wv" pathLength="1" style="stroke:url(#w{g});opacity:{op};stroke-width:{sw}" d="{smooth_path(P)}"/>'
    ticks = "".join(
        f'<line x1="8" x2="14" y1="{10 + (H - 20) * i / 8:.1f}" y2="{10 + (H - 20) * i / 8:.1f}" class="grid"/>'
        f'<line x1="{W - 14}" x2="{W - 8}" y1="{10 + (H - 20) * i / 8:.1f}" y2="{10 + (H - 20) * i / 8:.1f}" class="grid"/>'
        for i in range(9)
    )
    return squash(
        f"""
        <div class="inset wave"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Probabilitas per prediksi">
        <defs><linearGradient id="wd" x1="0" x2="1"><stop offset=".1" stop-color="#22D3EE"/><stop offset=".9" stop-color="#3c2cff"/></linearGradient>
        <linearGradient id="wm" x1="0" x2="1"><stop offset=".15" stop-color="#FFB800"/><stop offset=".85" stop-color="#FF3DA5"/></linearGradient></defs>
        {ticks}<line class="grid" x1="{pad + 6}" x2="{W - pad - 6}" y1="{mid}" y2="{mid}"/>{paths}</svg></div>
        """
    )


# ---------- treemap ----------
def squarify(items, x, y, w, h):
    total = sum(i["v"] for i in items) or 1
    scale = w * h / total
    rest = [dict(i, a=i["v"] * scale) for i in items]
    out = []

    def worst(row, length):
        s = sum(r["a"] for r in row)
        mx, mn = max(r["a"] for r in row), min(r["a"] for r in row)
        return max(length * length * mx / (s * s), s * s / (length * length * mn))

    while rest:
        length = min(w, h)
        row, i = [rest[0]], 1
        while i < len(rest) and worst(row + [rest[i]], length) <= worst(row, length):
            row.append(rest[i])
            i += 1
        rest = rest[i:]
        s = sum(r["a"] for r in row)
        if w >= h:
            cw, cy = s / h, y
            for r in row:
                rh = r["a"] / cw
                out.append(dict(r, x=x, y=cy, w=cw, h=rh))
                cy += rh
            x += cw
            w -= cw
        else:
            rh, cx = s / w, x
            for r in row:
                rw = r["a"] / rh
                out.append(dict(r, x=cx, y=y, w=rw, h=rh))
                cx += rw
            y += rh
            h -= rh
    return out


def treemap_html(rows: list) -> str:
    counts: dict[tuple, int] = {}
    for r in rows:
        k = (r["Model"], r["Hasil"])
        counts[k] = counts.get(k, 0) + 1
    if not counts:
        return placeholder("🧩", "Belum ada data", "Treemap muncul setelah prediksi pertama.")
    items = sorted(({"name": f"{m} · {c}", "cls": c, "v": n} for (m, c), n in counts.items()), key=lambda i: -i["v"])
    total = sum(i["v"] for i in items)
    VW, VH, PX_W, PX_H = 233.0, 100.0, 700.0, 300.0
    cells = ""
    for n, r in enumerate(squarify(items, 0, 0, VW, VH)):
        wpx, hpx = r["w"] / VW * PX_W, r["h"] / VH * PX_H
        share = f"{r['v'] / total * 100:.1f}".rstrip("0").rstrip(".") + "%"
        chars = len(str(r["v"]))
        fs = min(30, (wpx - 30 - 58) / (chars * 0.62))
        show_pct = True
        if fs < 17:
            show_pct, fs = False, min(30, (wpx - 30) / (chars * 0.62))
        tiny = fs < 12 or hpx < 54 or wpx < 60
        show_name = hpx >= 78 and wpx >= 96
        body = ""
        if not tiny:
            pill = f'<span class="tp">{share}</span>' if show_pct else ""
            body = f'<div class="tm-bot"><span class="ab-mono" style="font-size:{fs:.0f}px">{r["v"]}</span>{pill}</div>'
        name = f'<span class="tm-name">{html.escape(r["name"])}</span>' if show_name else ""
        cells += (
            f'<div class="tm-cell" style="left:{r["x"] / VW * 100:.3f}%;top:{r["y"] / VH * 100:.3f}%;'
            f'width:{r["w"] / VW * 100:.3f}%;height:{r["h"] / VH * 100:.3f}%;animation-delay:{n * 60}ms">'
            f'<div class="tile {class_of(r["cls"])}" title="{html.escape(r["name"])}: {r["v"]} ({share})">'
            f'<div class="tm-top">{ICON}{name}</div>{body}</div></div>'
        )
    return f'<div class="treemap">{cells}</div>'


# ---------- radar ----------
def radar_html(rows: list) -> str:
    vals = [r["Keyakinan (%)"] for r in reversed(rows)][-12:]
    N = len(vals)
    if N < 3:
        return placeholder("🕸️", "Radar belum tersedia", "Butuh minimal 3 prediksi untuk membentuk pola.")
    S, C = 420, 210
    R = S / 2 - 34
    lo, hi = min(vals), max(vals)
    nv = [0.42 + (v - lo) / ((hi - lo) or 1) * 0.58 for v in vals]
    K = min(5, N // 2)
    co = []
    for k in range(K + 1):
        a = sum(x * np.cos(k * (i / N) * 2 * np.pi) for i, x in enumerate(nv))
        b = sum(x * np.sin(k * (i / N) * 2 * np.pi) for i, x in enumerate(nv))
        f = 2 if k else 1
        co.append((a * f / N, b * f / N))
    fit = lambda th: sum(a * np.cos(k * th) + b * np.sin(k * th) for k, (a, b) in enumerate(co))

    def at(th):
        v = max(0.05, fit(th))
        ang = -np.pi / 2 + th
        r = R * 0.2 + v * R * 0.8
        return C + np.cos(ang) * r, C + np.sin(ang) * r

    d = ""
    for j in range(181):
        x, y = at(j / 180 * 2 * np.pi)
        d += ("L" if j else "M") + f"{x:.1f},{y:.1f}"
    d += "Z"
    ang = lambda i: -np.pi / 2 + i / N * 2 * np.pi
    circles = "".join(f'<circle class="rg" cx="{C}" cy="{C}" r="{R * k:.1f}"/>' for k in (0.33, 0.66, 1))
    axes = ""
    for i in range(N):
        a = ang(i)
        axes += (
            f'<line class="rg" x1="{C + np.cos(a) * R * 0.2:.1f}" y1="{C + np.sin(a) * R * 0.2:.1f}" '
            f'x2="{C + np.cos(a) * R:.1f}" y2="{C + np.sin(a) * R:.1f}"/>'
            f'<text class="rl" x="{C + np.cos(a) * (R + 18):.1f}" y="{C + np.sin(a) * (R + 18):.1f}" '
            f'text-anchor="middle" dominant-baseline="central">#{i + 1}</text>'
        )
    pts = ""
    for i, v in enumerate(vals):
        x, y = at(i / N * 2 * np.pi)
        pts += f'<g class="pt"><circle r="12" cx="{x:.1f}" cy="{y:.1f}" fill="transparent"/><circle class="dot" r="4.5" cx="{x:.1f}" cy="{y:.1f}"/><title>Prediksi #{i + 1}: {v:.1f}%</title></g>'
    hexa = (
        f'<g transform="translate({C - 11} {C - 11})" fill="none" class="hx"><path d="M11 1.5l8.5 5v9L11 20.5l-8.5-5v-9z '
        f'M11 1.5v19 M2.5 6.5L19.5 15.5 M19.5 6.5L2.5 15.5"/></g>'
    )
    return squash(
        f"""
        <div class="radar"><svg viewBox="0 0 {S} {S}" role="img" aria-label="Pola keyakinan prediksi">
        {circles}{axes}<path class="rline" pathLength="1" d="{d}"/>
        <circle class="rc" cx="{C}" cy="{C}" r="{R * 0.2:.1f}"/>{hexa}{pts}</svg></div>
        """
    )


# ---------- tabel ----------
def table_html(rows: list) -> str:
    models: dict[str, list] = {}
    for r in rows:
        models.setdefault(r["Model"], []).append(r)
    if not models:
        return ""
    body = ""
    for m, rs in models.items():
        body += (
            f'<tr><td class="l">{html.escape(m)}</td><td>{len(rs)}</td>'
            f'<td>{np.mean([r["Keyakinan (%)"] for r in rs]):.1f}%</td>'
            f'<td>{np.mean([r["Waktu (ms)"] for r in rs]):.0f} ms</td></tr>'
        )
    return (
        '<table class="ab-table"><thead><tr><th class="l">Model</th><th>Total</th><th>Keyakinan</th><th>Waktu</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


# ---------- dashboard (4 kartu bento) ----------
def dashboard_html(rows: list, threshold: float) -> str:
    n = len(rows)
    legend_cls = [c for c in CLASS_NAMES]
    present = {r["Hasil"] for r in rows}
    legend_cls += sorted(present - set(CLASS_NAMES))
    legend = "".join(
        f'<span><i class="sw {class_of(c)}"></i>{html.escape(c)}</span>' for c in legend_cls
    )

    if n:
        confs = [r["Keyakinan (%)"] for r in rows]
        last = confs[0]
        avg = float(np.mean(confs))
        avg_ms = float(np.mean([r["Waktu (ms)"] for r in rows]))
        above = sum(c >= threshold * 100 for c in confs) / n * 100
        sources = len({r["Sumber"] for r in rows})
        d_last = last - threshold * 100
        d_avg = avg - threshold * 100
        c1 = squash(
            f"""
            <div class="bcard span5"><div class="top-card"><div class="top-main">
            <div class="ab-label">Keyakinan terakhir</div>
            <div class="ab-big">{count_up(last, '%')}{delta_html(d_last, f'{abs(d_last):.1f}')}</div>
            <div class="ab-note">Rata-rata <b>{avg:.1f}%</b></div></div>{spark_inset(rows, 'a')}</div></div>
            """
        )
        c2 = squash(
            f"""
            <div class="bcard span7"><div class="top-card"><div class="top-main">
            <div class="ab-label">Total prediksi</div>
            <div class="ab-big">{n:,}{delta_html(above - 50, f'{above:.0f}%')}</div>
            <div class="ab-note">Per prediksi, ambang {threshold * 100:.0f}%</div>
            <div class="legend2"><span><i class="ln up"></i>{html.escape(CLASS_NAMES[0])}</span>
            <span><i class="ln down"></i>{html.escape(CLASS_NAMES[1])}</span><span><i class="ln prior"></i>Model kedua</span></div></div>
            {waves_html(rows)}</div></div>
            """
        )
        periods = squash(
            f"""
            <div class="periods">
            <div><div class="pl"><i class="on"></i>Total</div><div class="ab-big md">{n}</div><div class="ab-note">dari {sources} sumber gambar</div></div>
            <div><div class="pl"><i></i>Keyakinan</div><div class="ab-big md">{avg:.1f}<small>%</small>{delta_html(d_avg, f'{abs(d_avg):.1f}')}</div><div class="ab-note">ambang {threshold * 100:.0f}%</div></div>
            <div><div class="pl"><i></i>Waktu</div><div class="ab-big md">{avg_ms:.0f}<small> ms</small></div><div class="ab-note">rata-rata per prediksi</div></div>
            </div>
            """
        )
        mx, mnv = max(confs), min(confs)
        cls_rows = ""
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["Hasil"]] = counts.get(r["Hasil"], 0) + 1
        for c, k in sorted(counts.items(), key=lambda kv: -kv[1]):
            cls_rows += f'<div class="ab-foot"><span>{EMOJI.get(c, "")} {html.escape(c)}</span><b>{k} ({k / n * 100:.0f}%)</b></div>'
        stats = squash(
            f"""
            <div class="two"><div><div class="ab-label">Keyakinan tertinggi</div><div class="ab-big md">{mx:.1f}<small>%</small></div>
            <div class="ab-note">{delta_html(mx - threshold * 100, f'{abs(mx - threshold * 100):.1f}')}</div></div>
            <div><div class="ab-label">Keyakinan terendah</div><div class="ab-big md">{mnv:.1f}<small>%</small></div>
            <div class="ab-note">{delta_html(mnv - threshold * 100, f'{abs(mnv - threshold * 100):.1f}')}</div></div></div>{cls_rows}
            """
        )
    else:
        empty_inset = '<div class="inset spark"><div class="inset-empty">Belum ada data</div></div>'
        c1 = squash(
            f"""
            <div class="bcard span5"><div class="top-card"><div class="top-main">
            <div class="ab-label">Keyakinan terakhir</div><div class="ab-big">–</div>
            <div class="ab-note">Lakukan prediksi pertama</div></div>{empty_inset}</div></div>
            """
        )
        c2 = squash(
            f"""
            <div class="bcard span7"><div class="top-card"><div class="top-main">
            <div class="ab-label">Total prediksi</div><div class="ab-big">0</div>
            <div class="ab-note">Probabilitas tiap prediksi</div></div>{waves_html([])}</div></div>
            """
        )
        periods = ""
        stats = ""

    card3 = squash(
        f"""
        <div class="bcard span7"><div class="card-head"><h3>Distribusi prediksi</h3><div class="legend">{legend}</div></div>
        {periods}{treemap_html(rows)}{table_html(rows)}</div>
        """
    )
    card4 = squash(
        f"""
        <div class="bcard span5"><div class="card-head"><h3>Pola keyakinan</h3><span class="ab-pill sm">12 prediksi terakhir</span></div>
        {radar_html(rows)}{stats}</div>
        """
    )
    return f'<div class="bento">{c1}{c2}</div><div class="bento">{card3}{card4}</div>'


def add_history(source: str, res: dict, label: str, conf: float):
    st.session_state.history.insert(
        0,
        {
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            "Sumber": source,
            "Model": res["name"],
            "Hasil": label,
            "Keyakinan (%)": round(conf * 100, 2),
            "Waktu (ms)": round(res["ms"]),
            "_p": [float(p) for p in res["probs"]],
        },
    )


def run_models(img: Image.Image, mode: str) -> list:
    if mode == "Bandingkan":
        return [predict_yolo(img), predict_keras(img)]  # YOLO dulu, baru TensorFlow
    if mode == "YOLOv8":
        return [predict_yolo(img)]
    return [predict_keras(img)]


def top_label(res: dict) -> tuple[str, float]:
    i = int(np.argmax(res["probs"]))
    return str(res["labels"][i]), float(res["probs"][i])


# ==========================================================
# State
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_key" not in st.session_state:
    st.session_state.last_key = None

# ==========================================================
# Header
# ==========================================================
st.markdown(
    squash(
        """
        <div class="ab-head"><div><h1>Horses or Humans</h1>
        <p>Klasifikasi gambar dengan CNN Keras dan YOLOv8. Unggah foto, ambil dari kamera, atau pilih contoh.</p></div>
        <span class="ab-pill"><i></i>2 kelas · 224 px</span></div>
        """
    ),
    unsafe_allow_html=True,
)

# ==========================================================
# Kontrol
# ==========================================================
with st.container(key="card-control"):
    c1, c2, c3 = st.columns([1.3, 1.1, 1.2], gap="large")
    with c1:
        st.markdown('<div class="ab-label">Model</div>', unsafe_allow_html=True)
        mode = st.segmented_control("Model", MODES, default=MODES[0], key="mode", label_visibility="collapsed") or MODES[0]
    with c2:
        st.markdown('<div class="ab-label">Sumber gambar</div>', unsafe_allow_html=True)
        source_type = st.segmented_control("Sumber", SOURCES, default=SOURCES[0], key="source", label_visibility="collapsed") or SOURCES[0]
    with c3:
        threshold = st.slider("Ambang keyakinan", 0.5, 0.99, 0.80, 0.01)

# ==========================================================
# Gambar + hasil
# ==========================================================
left, right = st.columns([5, 7], gap="medium")
img, source_name = None, ""

with left:
    with st.container(key="card-image"):
        st.markdown('<div class="ab-title">Gambar</div>', unsafe_allow_html=True)
        if source_type == "Unggah":
            up = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if up is not None:
                img, source_name = Image.open(up).convert("RGB"), up.name
        elif source_type == "Kamera":
            cam = st.camera_input("Ambil foto", label_visibility="collapsed")
            if cam is not None:
                img, source_name = Image.open(cam).convert("RGB"), "kamera"
        else:
            samples = (
                sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in IMG_TYPES) if SAMPLE_DIR.exists() else []
            )
            if not samples:
                st.markdown(
                    placeholder("📁", "Folder sample_image kosong", "Tambahkan beberapa gambar ke folder tersebut di repo."),
                    unsafe_allow_html=True,
                )
            else:
                choice = st.selectbox("Pilih gambar contoh", [p.name for p in samples], label_visibility="collapsed")
                img, source_name = Image.open(SAMPLE_DIR / choice).convert("RGB"), choice
                if st.button("Analisis semua contoh", help="Menjalankan model pada semua gambar contoh (maks. 24) agar dashboard terisi."):
                    bar = st.progress(0.0, text="Menganalisis...")
                    batch = samples[:24]
                    for i, p in enumerate(batch):
                        im = Image.open(p).convert("RGB")
                        for res in run_models(im, mode):
                            lab, cf = top_label(res)
                            add_history(p.name, res, lab, cf)
                        bar.progress((i + 1) / len(batch), text=f"Menganalisis {i + 1}/{len(batch)}")
                    bar.empty()
                    st.toast(f"{len(batch)} gambar selesai dianalisis", icon="✅")
        if img is not None:
            st.image(img, caption=source_name, width="stretch")

with right:
    with st.container(key="card-result"):
        st.markdown('<div class="ab-title">Hasil prediksi</div>', unsafe_allow_html=True)
        if img is None:
            st.markdown(
                placeholder("🖼️", "Belum ada gambar", "Pilih sumber gambar lalu unggah atau pilih gambar untuk memulai."),
                unsafe_allow_html=True,
            )
        else:
            key = (hashlib.md5(img.tobytes()).hexdigest(), mode)
            is_new = key != st.session_state.last_key
            with st.spinner("Menganalisis gambar..."):
                results = run_models(img, mode)

            outputs = []
            cols = st.columns(len(results), gap="medium")
            for col, res in zip(cols, results):
                block, label, conf = result_html(res, threshold)
                outputs.append((label, conf))
                with col:
                    st.markdown(block, unsafe_allow_html=True)

            if len(results) == 2:
                if outputs[0][0] == outputs[1][0]:
                    st.markdown(
                        f'<div class="ab-ok">Kedua model sepakat: {html.escape(outputs[0][0])} {EMOJI.get(outputs[0][0], "")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ab-warn">Kedua model memberi hasil berbeda. Coba gambar yang lebih jelas.</div>',
                        unsafe_allow_html=True,
                    )

            if is_new:
                for res, (label, conf) in zip(results, outputs):
                    add_history(source_name, res, label, conf)
                st.session_state.last_key = key
                if all(c >= 0.95 for _, c in outputs):
                    st.toast("Keyakinan sangat tinggi!", icon="🎯")

# ==========================================================
# Dashboard bento
# ==========================================================
history = st.session_state.history
st.write("")
models_present = sorted({r["Model"] for r in history})
flt = "Semua"
if len(models_present) > 1:
    fl, _ = st.columns([2, 3])
    with fl:
        flt = st.segmented_control("Filter model", ["Semua"] + models_present, default="Semua", label_visibility="collapsed") or "Semua"
rows = history if flt == "Semua" else [r for r in history if r["Model"] == flt]

st.markdown(dashboard_html(rows, threshold), unsafe_allow_html=True)

# ==========================================================
# Riwayat lengkap
# ==========================================================
with st.container(key="card-history"):
    st.markdown('<div class="ab-title">Riwayat prediksi</div>', unsafe_allow_html=True)
    if not history:
        st.markdown(placeholder("🕑", "Belum ada riwayat", "Hasil prediksi akan muncul di sini."), unsafe_allow_html=True)
    else:
        hist = pd.DataFrame(history).drop(columns=["_p"])
        st.dataframe(hist, width="stretch", hide_index=True, height=240)
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button("Unduh CSV", hist.to_csv(index=False).encode("utf-8"), "riwayat_prediksi.csv", "text/csv")
        if b2.button("Hapus riwayat"):
            st.session_state.history = []
            st.rerun()