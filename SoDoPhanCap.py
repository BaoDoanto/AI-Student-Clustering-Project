"""
🎓 Phần mềm Phân nhóm Sinh viên – Hierarchical Clustering
Báo cáo: Áp dụng thuật toán gom cụm phân cấp vào bài toán gom cụm sinh viên
Dữ liệu: 60 sinh viên · 4 cụm có nhãn ngữ nghĩa
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import io

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân nhóm Sinh viên – Hierarchical Clustering",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
.main-title {
    background: linear-gradient(135deg,#1a3a6c 0%,#2563eb 60%,#0ea5e9 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:2.1rem;font-weight:800;line-height:1.2;
}
.subtitle{color:#64748b;font-size:.95rem;margin-bottom:6px}
.cluster-legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.cl-badge{padding:6px 14px;border-radius:20px;font-size:.83rem;font-weight:700;border:2px solid;display:inline-block}
.cl-1{background:#fef9c3;color:#854d0e;border-color:#fbbf24}
.cl-2{background:#dbeafe;color:#1e40af;border-color:#3b82f6}
.cl-3{background:#dcfce7;color:#166534;border-color:#22c55e}
.cl-4{background:#ede9fe;color:#5b21b6;border-color:#8b5cf6}
.stat-box{background:#f8faff;border:1px solid #dbeafe;border-left:4px solid;border-radius:8px;padding:12px 16px;margin-bottom:10px}
.stat-box h4{margin:0 0 4px 0;font-size:.9rem;font-weight:700}
.stat-box p{margin:0;font-size:.82rem;color:#475569;line-height:1.6}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DỮ LIỆU GỐC 60 SINH VIÊN
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA = {
    'Sinh viên': [f"SV{i+1:02d}" for i in range(60)],
    'GPA': [
        3.24,3.31,3.32,3.11,2.42,3.01,3.37,2.36,3.19,2.94,
        2.59,2.82,2.95,2.36,3.42,2.19,3.13,2.50,2.57,3.28,
        2.98,3.38,2.72,3.35,2.67,2.75,3.54,3.18,3.33,3.64,
        2.79,3.57,3.04,2.59,2.15,2.69,2.68,2.83,2.80,3.30,
        2.41,2.79,2.99,3.04,3.13,3.38,2.73,2.45,2.85,3.55,
        2.57,3.22,3.00,3.09,3.22,2.57,3.85,2.77,3.68,3.50,
    ],
    'Diem_RL': [
         70, 75, 73, 79, 50, 75, 84, 70, 76, 90,
         70, 69, 70, 65, 70, 64, 73, 80,107,105,
         82, 88, 70, 88, 83, 70, 83, 73, 76, 73,
         78, 83, 70, 68, 78, 70, 78, 73, 82, 73,
         69, 69, 92, 78, 70, 83, 73, 69, 74, 81,
         70,109, 74, 79, 74, 69, 92, 70,106, 91,
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH 4 NHÓM
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_CONFIG = {
    "Hoc Ba":     {"label": "Học Bá",       "emoji": "🏆", "color": "#8b5cf6", "bg": "#ede9fe", "border": "#8b5cf6"},
    "Hoc Thuat":  {"label": "Học Thuật",    "emoji": "📚", "color": "#22c55e", "bg": "#dcfce7", "border": "#22c55e"},
    "Nang No":    {"label": "Năng Nổ",      "emoji": "⭐", "color": "#3b82f6", "bg": "#dbeafe", "border": "#3b82f6"},
    "Can Ho Tro": {"label": "Cần Hỗ Trợ",  "emoji": "⚠️", "color": "#f59e0b", "bg": "#fef9c3", "border": "#fbbf24"},
}
CLUSTER_KEYS   = list(CLUSTER_CONFIG.keys())
CLUSTER_LABELS = [CLUSTER_CONFIG[k]["label"] for k in CLUSTER_KEYS]
CLUSTER_COLORS = [CLUSTER_CONFIG[k]["color"] for k in CLUSTER_KEYS]


# ─────────────────────────────────────────────────────────────────────────────
# HÀM XẾP LOẠI NHÓM (ĐÃ SỬA)
# ─────────────────────────────────────────────────────────────────────────────
def assign_label(gpa, drl):
    """
    Quy tắc xếp nhóm ưu tiên từ cao xuống thấp:
    1. Học Bá: GPA xuất sắc VÀ ĐRL tốt -> Đủ điều kiện học bổng.
    2. Học Thuật: GPA giỏi, ĐRL ở mức khá.
    3. Năng Nổ: Hoạt động tốt, GPA mức trung bình/khá.
    4. Cần Hỗ Trợ: GPA quá thấp (<2.0) HOẶC ĐRL quá thấp (<70).
    """
    # 1. Nhóm Học Bá (Ưu tiên số 1)
    if gpa >= 3.5 and drl > 80:
        return "Hoc Ba"
    
    # 2. Nhóm Học Thuật (GPA cao nhưng ĐRL chưa tới mức Học Bá)
    elif gpa >= 3.2:
        # Lưu ý: Nếu ĐRL < 70 ở nhóm này vẫn nên nhắc nhở, 
        # nhưng xét về trình độ họ vẫn thuộc nhóm Học thuật/Khá.
        return "Hoc Thuat"
    
    # 3. Nhóm Năng Nổ (Hoạt động tốt, GPA ổn định)
    elif drl >= 70 and gpa >= 2.0:
        return "Nang No"
    
    # 4. Các trường hợp còn lại (GPA yếu hoặc ĐRL quá kém)
    else:
        return "Can Ho Tro"

# ─────────────────────────────────────────────────────────────────────────────
# HÀM KIỂM TRA TRƯỜNG HỢP ĐẶC BIỆT
# ─────────────────────────────────────────────────────────────────────────────
def is_special_case_hoc_thuat(gpa, drl):
    """
    Trả về True nếu GPA cực cao (>= 3.5) nhưng ĐRL chưa đạt mức Học Bá (70-80).
    """
    if gpa >= 3.5 and 70 <= drl < 80:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TIÊU ĐỀ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Phân nhóm Sinh viên bằng Hierarchical Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Báo cáo đề tài · Áp dụng thuật toán gom cụm phân cấp vào bài toán gom cụm sinh viên</div>', unsafe_allow_html=True)
st.markdown("""
<div class="cluster-legend">
  <span class="cl-badge cl-4">🏆 Học Bá – GPA ≥ 3.5 · ĐRL &gt; 80 (đủ học bổng)</span>
  <span class="cl-badge cl-3">📚 Học Thuật – GPA ≥ 3.2 · ĐRL ≥ 70</span>
  <span class="cl-badge cl-2">⭐ Năng Nổ – GPA 2.0–3.1 · ĐRL ≥ 70</span>
  <span class="cl-badge cl-1">⚠️ Cần Hỗ Trợ – GPA &lt; 2.0 hoặc ĐRL &lt; 70</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt thuật toán")
    method_map = {
        "Ward (Khuyến nghị)":         "ward",
        "Complete (Max khoảng cách)": "complete",
        "Single (Min khoảng cách)":   "single",
        "Average (TB khoảng cách)":   "average",
    }
    method_label = st.selectbox("Phương pháp liên kết", list(method_map.keys()))
    method       = method_map[method_label]
    orientation  = st.radio("Hướng Dendrogram", ["top", "left"],
                            format_func=lambda x: "Dọc (top→down)" if x == "top" else "Ngang (left→right)")
    show_labels  = st.checkbox("Hiện tên SV", value=True)

    st.markdown("---")
    st.markdown("**📂 Tải lên CSV**")
    uploaded = st.file_uploader("", type=["csv"],
                                help="Cần cột: Sinh viên · GPA · Điểm RL")

    st.markdown("---")
    with st.expander("ℹ️ Tiêu chí xếp nhóm"):
        st.markdown("""
| Nhóm | GPA | ĐRL | Ghi chú |
|------|-----|-----|---------|
| 🏆 Học Bá | ≥ 3.5 | > 80 | Xét học bổng KK |
| 📚 Học Thuật | ≥ 3.2 | ≥ 70 | GPA tốt |
| ⭐ Năng Nổ | ≥ 2.0 | ≥ 70 | HĐ tốt |
| ⚠️ Cần HT | < 2.0 | < 70 | Can thiệp kịp thời |

**Trường hợp đặc biệt:**
GPA ≥ 3.5 nhưng ĐRL 70–80 → Học Thuật + khuyến khích tham gia HĐ *(không bắt buộc)* để đủ chuẩn học bổng.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
def _load_csv(source):
    raw = pd.read_csv(source)
    raw.columns = [c.strip() for c in raw.columns]
    if 'Điểm RL' in raw.columns:
        raw = raw.rename(columns={'Điểm RL': 'Diem_RL'})
    if not {'Sinh viên', 'GPA', 'Diem_RL'}.issubset(raw.columns):
        raise ValueError("File thiếu cột bắt buộc!")
    return raw

if uploaded:
    try:
        df = _load_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Lỗi file upload: {e}")
        df = pd.DataFrame(RAW_DATA)
else:
    try:
        df = _load_csv("DuLieuSinhVien.csv")
    except Exception:
        df = pd.DataFrame(RAW_DATA)

df = df.copy()
labels_sv = df['Sinh viên'].tolist()
data      = df[['GPA', 'Diem_RL']].values
n         = len(labels_sv)

# GÁN NHÃN VÀ ĐÁNH DẤU (Dòng 201 trong file của bạn)
# ─────────────────────────────────────────────────────────────────────────────
df['NhomKey']   = df.apply(lambda r: assign_label(r['GPA'], r['Diem_RL']), axis=1)
df['NhomLabel'] = df['NhomKey'].map(lambda k: CLUSTER_CONFIG[k]['label'])

# Gọi hàm kiểm tra trường hợp đặc biệt
df['DacBiet']   = df.apply(lambda r: is_special_case_hoc_thuat(r['GPA'], r['Diem_RL']), axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
Z  = linkage(data, method=method)
k  = 4
hc = fcluster(Z, k, criterion='maxclust')
df['HC_Cluster'] = hc

sorted_dist = sorted(Z[:, 2])
idx   = max(n - k, 0)
cut_h = (sorted_dist[idx - 1] + sorted_dist[idx]) / 2 if idx > 0 else sorted_dist[-1] * 0.5

LEAF_COLOR = {sv: CLUSTER_CONFIG[gk]["color"]
              for sv, gk in zip(labels_sv, df['NhomKey'])}

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT: Data | Dendrogram
# ─────────────────────────────────────────────────────────────────────────────
col_data, col_dend = st.columns([1, 2.2])

with col_data:
    st.markdown("#### 📋 Dữ liệu 60 sinh viên")
    GROUP_BG = {k: CLUSTER_CONFIG[k]["bg"] for k in CLUSTER_KEYS}

    display = df[['Sinh viên', 'GPA', 'Diem_RL', 'NhomLabel']].rename(
        columns={'Diem_RL': 'Điểm RL', 'NhomLabel': 'Nhóm'})
    styled = display.style.apply(
        lambda row: [f'background-color:{GROUP_BG.get(df.loc[row.name, "NhomKey"], "#fff")}'] * len(row),
        axis=1
    ).format({'GPA': '{:.2f}'})
    st.dataframe(styled, use_container_width=True, height=380)

    st.markdown("#### 📊 Thống kê theo nhóm")
    for k_key in CLUSTER_KEYS:
        cfg = CLUSTER_CONFIG[k_key]
        sub = df[df['NhomKey'] == k_key]
        if sub.empty:
            continue
        cnt  = len(sub)
        svs  = "  ·  ".join(sub['Sinh viên'].tolist())
        agpa = sub['GPA'].mean()
        arl  = sub['Diem_RL'].mean()
        bc   = cfg['border']
        # Đánh dấu trường hợp đặc biệt trong nhóm Học Thuật
        extra = ""
        if k_key == "Hoc Thuat":
            dac_biet = sub[sub['DacBiet'] == True]
            if not dac_biet.empty:
                db_sv = " · ".join(dac_biet['Sinh viên'].tolist())
                extra = f"<p style='font-size:.76rem;color:#854d0e'>⭐ GPA ≥ 3.5 nhưng ĐRL 70–80 (cần thêm HĐ): {db_sv}</p>"
        st.markdown(f"""
<div class="stat-box" style="border-left-color:{bc}">
  <h4 style="color:{bc}">{cfg['emoji']} {cfg['label']} <span style="font-weight:400;font-size:.78rem;color:#94a3b8">({cnt} SV)</span></h4>
  <p>GPA TB: <b>{agpa:.2f}</b> &nbsp;·&nbsp; ĐRL TB: <b>{arl:.1f}</b></p>
  <p style="font-size:.76rem;color:#64748b;word-break:break-all">{svs}</p>
  {extra}
</div>""", unsafe_allow_html=True)

with col_dend:
    st.markdown("#### 🌳 Sơ đồ Phân cấp – Dendrogram")

    fw = 16 if orientation == "top" else 10
    fh =  8 if orientation == "top" else 16
    fig_d, ax_d = plt.subplots(figsize=(fw, fh))
    fig_d.patch.set_facecolor('#f8faff')
    ax_d.set_facecolor('#f8faff')

    ddata = dendrogram(
        Z,
        labels=labels_sv if show_labels else None,
        orientation=orientation,
        leaf_font_size=7,
        leaf_rotation=90 if orientation == "top" else 0,
        color_threshold=cut_h,
        above_threshold_color="#94a3b8",
        ax=ax_d,
    )

    if show_labels:
        ticks = ax_d.get_xticklabels() if orientation == "top" else ax_d.get_yticklabels()
        for tick in ticks:
            sv = tick.get_text()
            tick.set_color(LEAF_COLOR.get(sv, "#334155"))
            tick.set_fontweight("bold")

    lkw = dict(color='#ef4444', linestyle='--', linewidth=2, alpha=0.85,
               label="Ngưỡng cắt → k=4")
    if orientation == "top":
        ax_d.axhline(y=cut_h, **lkw)
        ax_d.set_ylabel("Khoảng cách liên kết", fontsize=10, color='#334155')
        ax_d.set_xlabel("Sinh viên", fontsize=10, color='#334155')
    else:
        ax_d.axvline(x=cut_h, **lkw)
        ax_d.set_xlabel("Khoảng cách liên kết", fontsize=10, color='#334155')
        ax_d.set_ylabel("Sinh viên", fontsize=10, color='#334155')

    patches = [
        mpatches.Patch(color=CLUSTER_CONFIG[k]["color"],
                       label=f"{CLUSTER_CONFIG[k]['emoji']} {CLUSTER_CONFIG[k]['label']}")
        for k in CLUSTER_KEYS
    ] + [mpatches.Patch(color='#ef4444', label="Ngưỡng cắt (k=4)")]
    ax_d.legend(handles=patches, loc='upper right', fontsize=9,
                framealpha=0.92, fancybox=True, shadow=True)

    ax_d.set_title(
        f"Dendrogram – Gom cụm phân cấp 60 sinh viên\n"
        f"Phương pháp: {method_label.split('(')[0].strip()} · 4 nhóm",
        fontsize=13, fontweight='bold', color='#1e40af', pad=14)
    ax_d.tick_params(colors='#475569')
    ax_d.spines[['top', 'right']].set_visible(False)
    ax_d.spines[['left', 'bottom']].set_color('#cbd5e1')
    plt.tight_layout()
    st.pyplot(fig_d)

    st.info(
        f"📌 **Đọc Dendrogram:** Màu lá = nhóm của sinh viên. "
        f"Chiều cao nhánh = mức độ dị biệt khi 2 cụm gộp. "
        f"**Đường đỏ** tại ngưỡng **{cut_h:.3f}** → **4 nhóm** phân biệt."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS PHÂN TÍCH BỔ SUNG
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Phân tích bổ sung")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Scatter Plot", "📊 Biểu đồ cột", "📐 Ma trận khoảng cách", "🌡️ Heatmap",
    "📋 Báo cáo Hỗ trợ Đào tạo"
])

with tab1:
    fig_sc, ax_sc = plt.subplots(figsize=(13, 6))
    fig_sc.patch.set_facecolor('#f8faff'); ax_sc.set_facecolor('#f8faff')
    for k_key in CLUSTER_KEYS:
        cfg = CLUSTER_CONFIG[k_key]
        sub = df[df['NhomKey'] == k_key]
        if sub.empty: continue
        ax_sc.scatter(sub['Diem_RL'], sub['GPA'],
                      color=cfg['color'], s=160, zorder=3,
                      edgecolors='white', linewidths=1.5,
                      label=f"{cfg['emoji']} {cfg['label']} ({len(sub)})")
        for _, row in sub.iterrows():
            ax_sc.annotate(row['Sinh viên'], (row['Diem_RL'], row['GPA']),
                           textcoords="offset points", xytext=(5, 3),
                           fontsize=7.5, color='#334155')
    # Đường phân vùng
    ax_sc.axhline(y=3.5, color='#8b5cf6', linestyle='--', lw=1.5, alpha=0.7, label='GPA = 3.5 (Học Bá)')
    ax_sc.axhline(y=3.2, color='#22c55e', linestyle=':', lw=1.3, alpha=0.7, label='GPA = 3.2 (Học Thuật)')
    ax_sc.axhline(y=2.0, color='#f59e0b', linestyle=':', lw=1.3, alpha=0.7, label='GPA = 2.0')
    ax_sc.axvline(x=80,  color='#8b5cf6', linestyle='--', lw=1.5, alpha=0.7, label='ĐRL = 80 (Học Bá)')
    ax_sc.axvline(x=70,  color='#f59e0b', linestyle=':', lw=1.3, alpha=0.7, label='ĐRL = 70')
    ax_sc.set_xlabel("Điểm Rèn luyện (ĐRL)", fontsize=11, color='#334155')
    ax_sc.set_ylabel("GPA", fontsize=11, color='#334155')
    ax_sc.set_title("Ma trận Sinh viên: Năng lực học tập × Hoạt động rèn luyện",
                    fontsize=12, fontweight='bold', color='#1e40af', pad=12)
    ax_sc.legend(fontsize=8.5, framealpha=0.9, fancybox=True, ncol=2)
    ax_sc.spines[['top', 'right']].set_visible(False)
    ax_sc.tick_params(colors='#475569')
    plt.tight_layout(); st.pyplot(fig_sc)

with tab2:
    groups_present = [k for k in CLUSTER_KEYS if not df[df['NhomKey'] == k].empty]
    clrs  = [CLUSTER_CONFIG[k]["color"] for k in groups_present]
    glbls = [f"{CLUSTER_CONFIG[k]['emoji']} {CLUSTER_CONFIG[k]['label']}" for k in groups_present]
    c1b, c2b = st.columns(2)
    with c1b:
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        fig_b.patch.set_facecolor('#f8faff'); ax_b.set_facecolor('#f8faff')
        gpas = [df[df['NhomKey'] == k]['GPA'].mean() for k in groups_present]
        bars = ax_b.bar(glbls, gpas, color=clrs, edgecolor='white', lw=1.5, zorder=3)
        ax_b.bar_label(bars, fmt='%.2f', padding=3, fontsize=11, fontweight='bold')
        for y, c in [(3.5, '#8b5cf6'), (3.2, '#22c55e')]:
            ax_b.axhline(y=y, color=c, linestyle='--', lw=1, alpha=0.6)
        ax_b.set_ylim(0, 4.3); ax_b.set_title("GPA trung bình", fontweight='bold', color='#1e40af')
        ax_b.set_ylabel("GPA"); ax_b.spines[['top', 'right']].set_visible(False)
        ax_b.grid(axis='y', alpha=0.3, zorder=0); ax_b.tick_params(colors='#475569')
        plt.xticks(rotation=15); plt.tight_layout(); st.pyplot(fig_b)
    with c2b:
        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        fig_r.patch.set_facecolor('#f8faff'); ax_r.set_facecolor('#f8faff')
        rls  = [df[df['NhomKey'] == k]['Diem_RL'].mean() for k in groups_present]
        bars2 = ax_r.bar(glbls, rls, color=clrs, edgecolor='white', lw=1.5, zorder=3)
        ax_r.bar_label(bars2, fmt='%.1f', padding=3, fontsize=11, fontweight='bold')
        for y, c in [(80, '#8b5cf6'), (70, '#f59e0b')]:
            ax_r.axhline(y=y, color=c, linestyle='--', lw=1, alpha=0.6)
        ax_r.set_ylim(0, 125); ax_r.set_title("ĐRL trung bình", fontweight='bold', color='#1e40af')
        ax_r.set_ylabel("Điểm Rèn luyện"); ax_r.spines[['top', 'right']].set_visible(False)
        ax_r.grid(axis='y', alpha=0.3, zorder=0); ax_r.tick_params(colors='#475569')
        plt.xticks(rotation=15); plt.tight_layout(); st.pyplot(fig_r)

    cp1, _, cp3 = st.columns([1, 0.3, 1])
    sizes = [len(df[df['NhomKey'] == k]) for k in groups_present]
    with cp1:
        fig_p, ax_p = plt.subplots(figsize=(5.5, 5))
        ax_p.pie(sizes,
                 labels=[f"{CLUSTER_CONFIG[k]['emoji']} {CLUSTER_CONFIG[k]['label']}\n({s} SV)"
                         for k, s in zip(groups_present, sizes)],
                 colors=clrs, autopct='%1.1f%%', startangle=140,
                 wedgeprops=dict(edgecolor='white', linewidth=2))
        ax_p.set_title("Tỷ lệ sinh viên theo nhóm", fontweight='bold', color='#1e40af')
        st.pyplot(fig_p)
    with cp3:
        st.markdown("**Số lượng sinh viên mỗi nhóm:**")
        for k_key, s in zip(groups_present, sizes):
            cfg = CLUSTER_CONFIG[k_key]
            st.markdown(
                f"<span style='color:{cfg['color']};font-weight:700'>"
                f"{cfg['emoji']} {cfg['label']}</span>: **{s} SV** ({s / n * 100:.1f}%)",
                unsafe_allow_html=True)

with tab3:
    n_show = min(20, n)
    dm     = squareform(pdist(data[:n_show], metric='euclidean'))
    df_dm  = pd.DataFrame(dm, index=labels_sv[:n_show], columns=labels_sv[:n_show])
    st.markdown(f"**Ma trận khoảng cách Euclidean – {n_show} sinh viên đầu**")
    st.dataframe(df_dm.style.format("{:.2f}").background_gradient(cmap='Blues', axis=None),
                 use_container_width=True)
    st.caption("Ô màu nhạt = khoảng cách nhỏ = tương đồng cao. HC gộp cặp nhỏ nhất trước.")

with tab4:
    n_hm  = min(30, n)
    order = df.sort_values('NhomKey')['Sinh viên'].tolist()[:n_hm]
    dm_a  = squareform(pdist(data, metric='euclidean'))
    idx_o = [labels_sv.index(l) for l in order]
    dm_s  = dm_a[np.ix_(idx_o, idx_o)]
    fig_hm, ax_hm = plt.subplots(figsize=(12, 9))
    fig_hm.patch.set_facecolor('#f8faff')
    sns.heatmap(dm_s, annot=True, fmt=".1f",
                xticklabels=order, yticklabels=order,
                cmap="YlOrRd", linewidths=0.4, linecolor='#e2e8f0',
                ax=ax_hm, annot_kws={"size": 7},
                cbar_kws={"label": "Khoảng cách Euclidean"})
    ax_hm.set_title(f"Heatmap khoảng cách ({n_hm} SV – sắp theo nhóm)",
                    fontsize=12, fontweight='bold', color='#1e40af', pad=12)
    plt.tight_layout(); st.pyplot(fig_hm)
    st.caption("Khối màu nhạt trên đường chéo = sinh viên cùng nhóm tương đồng nhau.")


# ── Tab 5: TỔNG KẾT HỖ TRỢ ĐÀO TẠO ─────────────────────────────────────────
with tab5:
    st.markdown("#### 📊 Tổng kết hành động cần thực hiện")

    actions = {
        "Hoc Ba":     ("🏆 Học Bá",       "Xét học bổng · Phân công Mentor · Đề cử NCKH",        "🟢 Duy trì & Phát triển"),
        "Hoc Thuat":  ("📚 Học Thuật",    "Yêu cầu Seminar · CLB chuyên môn · Phấn đấu Học Bá", "🟡 Theo dõi & Khuyến khích"),
        "Nang No":    ("⭐ Năng Nổ",      "Study Group · Gia sư đồng hành · Cân bằng thời gian", "🟡 Theo dõi & Hỗ trợ"),
        "Can Ho Tro": ("⚠️ Cần Hỗ Trợ",  "Cảnh báo học vụ · Tư vấn tâm lý · Hỗ trợ nhẹ",      "🔴 Ưu tiên cao"),
    }
    summary_data = {"Nhóm": [], "Số SV": [], "Danh sách sinh viên": [], "Hành động chính": [], "Mức độ ưu tiên": []}
    for k_key, (label, action, priority) in actions.items():
        sub = df[df['NhomKey'] == k_key]
        cnt = len(sub)
        if cnt > 0:
            summary_data["Nhóm"].append(label)
            summary_data["Số SV"].append(cnt)
            summary_data["Danh sách sinh viên"].append("  ·  ".join(sub['Sinh viên'].tolist()))
            summary_data["Hành động chính"].append(action)
            summary_data["Mức độ ưu tiên"].append(priority)

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Trường hợp đặc biệt
    db = df[df['DacBiet'] == True]
    if not db.empty:
        db_list = "  ·  ".join(db['Sinh viên'].tolist())
        st.warning(
            f"⭐ **Trường hợp đặc biệt ({len(db)} SV):** {db_list}\n\n"
            f"GPA ≥ 3.5 nhưng ĐRL 70–80. Khuyến khích tham gia thêm hoạt động *(không bắt buộc)* "
            f"để nâng ĐRL > 80, đủ điều kiện xét học bổng Học Bá."
        )

    # CSV
    out_csv = df[['Sinh viên', 'GPA', 'Diem_RL', 'NhomLabel']].rename(
        columns={'Diem_RL': 'Điểm RL', 'NhomLabel': 'Nhóm'}).to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Tải danh sách phân nhóm (CSV)",
                       out_csv, "bao_cao_ho_tro_dao_tao.csv", "text/csv",
                       use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# XUẤT KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⬇️ Xuất kết quả")
ec1, ec2, ec3 = st.columns(3)

with ec1:
    out_df = df[['Sinh viên', 'GPA', 'Diem_RL', 'NhomLabel']].rename(
        columns={'Diem_RL': 'Điểm RL', 'NhomLabel': 'Nhóm'})
    csv_b = out_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📄 CSV kết quả", csv_b,
                       "ket_qua_phan_nhom_60sv.csv", "text/csv",
                       use_container_width=True)

with ec2:
    buf = io.BytesIO()
    fig_d.savefig(buf, format="png", dpi=180, bbox_inches='tight', facecolor='#f8faff')
    buf.seek(0)
    st.download_button("🌳 PNG Dendrogram", buf,
                       "dendrogram_60sv.png", "image/png",
                       use_container_width=True)

with ec3:
    buf2 = io.BytesIO()
    fig_sc.savefig(buf2, format="png", dpi=180, bbox_inches='tight', facecolor='#f8faff')
    buf2.seek(0)
    st.download_button("🗺️ PNG Scatter Plot", buf2,
                       "scatter_60sv.png", "image/png",
                       use_container_width=True)

st.markdown("""
<div style="text-align:center;margin-top:24px;color:#94a3b8;font-size:.82rem">
  Đề tài: <b>Áp dụng thuật toán gom cụm phân cấp vào bài toán gom cụm sinh viên</b><br>
  60 sinh viên · 4 nhóm · Python · Streamlit · SciPy · Matplotlib · Seaborn
</div>
""", unsafe_allow_html=True)