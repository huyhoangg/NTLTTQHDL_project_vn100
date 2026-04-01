import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── VÁ LỖI TƯƠNG THÍCH PHIÊN BẢN (Numpy 2.0+ & Pandas 2.0+) ──
# Các thư viện cũ như dask, lightgbm, pandas bị gãy do tìm gọi các hàm Numpy/Pandas đời cũ đã bị xoá.
if not hasattr(np, 'unicode_'): np.unicode_ = str
if not hasattr(np, 'round_'): np.round_ = np.round
if not hasattr(np, 'float_'): np.float_ = float
if not hasattr(np, 'int_'): np.int_ = int
if not hasattr(np, 'bool_'): np.bool_ = bool
if not hasattr(np, 'object_'): np.object_ = object

try:
    if not hasattr(pd.core.strings, 'StringMethods'):
        pd.core.strings.StringMethods = pd.core.strings.accessor.StringMethods
except Exception:
    pass
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# ── Kết nối thư viện ModelAPI ────────────────────────────────────────────────
# Dùng Path(__file__).parent để lấy đường dẫn tương đối (chuẩn Deploy Cloud)
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
test_model_dir = current_dir / "ModelAPI" / "test_model"

if str(test_model_dir) not in sys.path:
    sys.path.insert(0, str(test_model_dir))

# Sẽ hiển thị lỗi rành mạch trên Streamlit nếu thiếu lightgbm
from model_apply_api import predict_symbols_from_master_csv, plot_single_symbol_direction, plot_latest_symbol_signal
st.set_page_config(page_title="VN100 Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Lưu cấu hình (Session State) ──────────────────────────────────────────────
if "w_ret" not in st.session_state: st.session_state.w_ret = 0.5
if "w_vol" not in st.session_state: st.session_state.w_vol = 0.3
if "w_volat" not in st.session_state: st.session_state.w_volat = 0.2
if "vol_spk" not in st.session_state: st.session_state.vol_spk = 1.5

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:var(--background-color)} .block-container{padding-top:1rem;padding-bottom:1rem;max-width:98%;} header{display:none!important;}
section[data-testid="stSidebar"]{background:var(--secondary-background-color);border-right:1px solid rgba(128,128,128,0.1)}
/* Nav radio styling */
div[data-testid="stSidebar"] .stRadio>div{gap:4px}
div[data-testid="stSidebar"] .stRadio label{
    display:block;width:100%;padding:10px 14px;border-radius:10px;
    cursor:pointer;color:var(--text-color);font-weight:500;font-size:.92rem;
    transition:all .15s;border:1px solid transparent;
}
div[data-testid="stSidebar"] .stRadio label:hover{background:rgba(128,128,128,0.1)}
div[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
div[data-testid="stSidebar"] .stRadio input:checked + div label{
    background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;border-color:#3b82f6
}
/* Metric cards */
div[data-testid="metric-container"]{
    background:var(--secondary-background-color);
    border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:8px 12px;
    box-shadow:0 4px 20px rgba(0,0,0,.08);transition:transform .18s,box-shadow .18s
}
div[data-testid="metric-container"]:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(37,99,235,.15)}
div[data-testid="metric-container"] label{color:#3b82f6!important;font-size:.76rem!important;font-weight:600!important}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:var(--text-color)!important;font-weight:700!important}
/* Control card in sidebar */
.ctrl-card{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:14px;margin-bottom:12px}
.ctrl-title{color:#3b82f6;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
/* Section header */
.sh{background:linear-gradient(90deg,rgba(37,99,235,.1),transparent);
    border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;
    padding:8px 14px;margin:16px 0 10px;color:var(--text-color);font-weight:600;font-size:1rem}
/* Selectbox, slider */
.stSelectbox>div>div,.stMultiSelect>div>div{
    background:var(--secondary-background-color)!important;border:1px solid rgba(128,128,128,0.2)!important;
    border-radius:8px!important;color:var(--text-color)!important}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:#3b82f6!important}
hr{border-color:rgba(128,128,128,0.2)!important;margin:.8rem 0!important}
h1,h2,h3{color:var(--text-color)!important}
/* Tabs (secondary within page) */
.stTabs [data-baseweb="tab-list"]{background:var(--secondary-background-color);border-radius:10px;padding:4px;gap:3px;border:1px solid rgba(128,128,128,0.2)}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:var(--text-color);font-size:.85rem;padding:6px 14px;opacity: 1; font-weight: 500;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1d4ed8,#6d28d9)!important;color:#fff!important;opacity: 1;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BG = "rgba(0,0,0,0)"

def pct_clr(v): return "#10b981" if v>0 else ("#ef4444" if v<0 else "#f59e0b")
def score_clr(s):
    if pd.isna(s): return "#9ca3af"
    if s<=-7: return "#06b6d4"
    if s<-3:  return "#ef4444"
    if s<0:   return "#fca5a5"
    if s==0:  return "#f59e0b"
    if s<3:   return "#10b981"
    if s<7:   return "#16a34a"
    return "#8b5cf6"

def dl(**kw):
    """Flexible layout preset supporting auto-theming font color"""
    return dict(
                font=dict(family="Inter"),
                margin=kw.pop('margin', dict(t=50,l=10,r=10,b=10)),
                **kw)

def fix_date_gaps(fig):
    """Hide non-trading days (weekends & holidays) while keeping date formatting"""
    full_range = pd.date_range(start=ALL_DATES[0], end=ALL_DATES[-1])
    gaps = full_range[~full_range.isin(ALL_DATES)]
    fig.update_xaxes(rangebreaks=[dict(values=gaps.strftime("%Y-%m-%d").tolist())])
    return fig

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Đang tải dữ liệu...")
def load_data():
    df = pd.read_csv("vn100_featured_data.csv", parse_dates=["date"])
    return df.sort_values(["ticker","date"]).reset_index(drop=True)

df = load_data()
ALL_DATES = sorted(pd.to_datetime(df["date"].unique()))
TICKERS   = sorted(df["ticker"].unique().tolist())
D1, D0    = ALL_DATES[-1], ALL_DATES[-2]

# ── Cached charts ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def kpi_data():
    d1 = df[df["date"]==D1]; d0 = df[df["date"]==D0]
    m = d1[["ticker","LS_GiaDongCua"]].merge(d0[["ticker","LS_GiaDongCua"]],on="ticker",suffixes=("_n","_p"))
    m["pct"]=(m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"])/m["LS_GiaDongCua_p"]*100
    return {
        "val":  d1["LS_GiaTriKhopLenh"].sum()/1e12,
        "vol":  d1["LS_KhoiLuongKhopLenh"].sum()/1e6,
        "up":   int((m["pct"]>0).sum()),
        "dn":   int((m["pct"]<0).sum()),
        "ref":  int((m["pct"]==0).sum()),
        "nn":   d1["KN_GTDGRong"].sum()/1e9 if "KN_GTDGRong" in d1 else 0,
        "nstk": d1["ticker"].nunique(),
    }

@st.cache_data(show_spinner=False)
def fig_treemap(top_n):
    d1 = df[df["date"]==D1][["ticker","icb_name2","LS_GiaDongCua","LS_GiaTriKhopLenh"]]
    d0 = df[df["date"]==D0][["ticker","LS_GiaDongCua"]]
    m = d1.merge(d0,on="ticker",suffixes=("_n","_p"))
    m["pct"]=(m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"])/m["LS_GiaDongCua_p"]*100
    ind = m.groupby("icb_name2").apply(lambda g: pd.Series({
        "val": g["LS_GiaTriKhopLenh"].sum(),
        "pct": (g["pct"]*g["LS_GiaTriKhopLenh"]).sum()/g["LS_GiaTriKhopLenh"].sum()
    }), include_groups=False).reset_index()
    ind = ind.sort_values("val",ascending=False).head(top_n)
    ind["tybnd"]=ind["val"]/1e9
    ind["color"]=ind["pct"].apply(lambda p: score_clr(p/1.0))
    fig = go.Figure(go.Treemap(
        labels=ind["icb_name2"],parents=[""]*len(ind),values=ind["tybnd"],
        text=[f"<b>{r.icb_name2}</b><br>{r.pct:+.2f}%" for _,r in ind.iterrows()],
        textinfo="text",textfont=dict(size=17,color="white"),textposition="middle center",
        marker=dict(colors=ind["color"],line=dict(color="rgba(128,128,128,0.2)",width=2)),
        hovertemplate="<b>%{label}</b><br>GTGD: %{value:,.0f} tỷ<extra></extra>",
        tiling=dict(pad=3,packing="squarify"),
    ))
    fig.update_layout(height=500,margin=dict(t=40,l=4,r=4,b=4),
        title=dict(text=f"🗺️ Bản đồ ngành — {D1.date()}",font=dict(size=18),x=0))
    return fig

@st.cache_data(show_spinner=False)
def fig_breadth():
    rows=[]
    for i in range(1,min(61,len(ALL_DATES))):
        d1,dp=ALL_DATES[-i],ALL_DATES[-i-1]
        n=df[df["date"]==d1][["ticker","LS_GiaDongCua"]]
        p=df[df["date"]==dp][["ticker","LS_GiaDongCua"]]
        m=n.merge(p,on="ticker",suffixes=("_n","_p"))
        diff=m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"]
        rows.append({"date":d1,"up":(diff>0).sum(),"dn":(diff<0).sum(),"ref":(diff==0).sum()})
    bd=pd.DataFrame(rows).sort_values("date")
    fig=go.Figure()
    fig.add_trace(go.Bar(x=bd["date"],y=bd["up"],  name="Tăng",    marker_color="#4ade80",opacity=.85))
    fig.add_trace(go.Bar(x=bd["date"],y=-bd["dn"], name="Giảm",    marker_color="#f87171",opacity=.85))
    fig.add_trace(go.Bar(x=bd["date"],y=bd["ref"], name="Tham chiếu",marker_color="#fbbf24",opacity=.7))
    fig.update_layout(barmode="relative",height=300,legend=dict(orientation="h",y=1.1),
        title=dict(text="📊 Độ rộng thị trường",font=dict(size=15),x=0),
        **dl(margin=dict(t=50,l=10,r=10,b=10)))
    return fix_date_gaps(fig)

@st.cache_data(show_spinner=False)
def fig_vol_trend():
    rows=[{"date":d,"val":df[df["date"]==d]["LS_GiaTriKhopLenh"].sum()/1e12}
          for d in ALL_DATES[-60:]]
    vt=pd.DataFrame(rows)
    ma=vt["val"].rolling(5).mean()
    fig=go.Figure()
    fig.add_trace(go.Bar(x=vt["date"],y=vt["val"],name="GTGD (nghìn tỷ)",marker_color="#3b82f6",opacity=.7))
    fig.add_trace(go.Scatter(x=vt["date"],y=ma,name="MA5",line=dict(color="#fbbf24",width=2)))
    fig.update_layout(height=270,legend=dict(),
        title=dict(text="💰 Giá trị GD toàn thị trường",font=dict(size=15),x=0),
        **dl(margin=dict(t=50,l=10,r=10,b=10)),yaxis_title="Nghìn tỷ")
    return fix_date_gaps(fig)



@st.cache_data(show_spinner=False)
def fig_sector_bar(period_days):
    if len(ALL_DATES)<=period_days: return None
    d1,dp=ALL_DATES[-1],ALL_DATES[-(period_days+1)]
    now=df[df["date"]==d1][["ticker","icb_name2","LS_GiaDongCua"]]
    prv=df[df["date"]==dp][["ticker","LS_GiaDongCua"]]
    m=now.merge(prv,on="ticker",suffixes=("_n","_p"))
    m["pct"]=(m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"])/m["LS_GiaDongCua_p"]*100
    sec=m.groupby("icb_name2")["pct"].mean().sort_values(ascending=True).reset_index()
    label={1:"1 ngày",7:"1 tuần",30:"1 tháng"}.get(period_days,f"{period_days} ngày")
    fig=go.Figure(go.Bar(x=sec["pct"],y=sec["icb_name2"],orientation="h",
        marker=dict(color=[pct_clr(v) for v in sec["pct"]],line_width=0),
        text=[f"{v:+.2f}%" for v in sec["pct"]],textposition="outside"))
    fig.update_layout(height=420,title=dict(text=f"📊 Hiệu suất ngành — {label}",font=dict(size=15),x=0),
        **dl(margin=dict(t=50,l=10,r=80,b=10)))
    return fig

@st.cache_data(show_spinner=False)
def fig_top_table(sort_col, is_value):
    d1=df[df["date"]==D1][["ticker","LS_GiaDongCua","LS_KhoiLuongKhopLenh","LS_GiaTriKhopLenh"]]
    d0=df[df["date"]==D0][["ticker","LS_GiaDongCua"]]
    m=d1.merge(d0,on="ticker",suffixes=("_n","_p"))
    m["pct"]=(m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"])/m["LS_GiaDongCua_p"]*100
    m["chg"]=m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"]
    top=m.sort_values(sort_col,ascending=False).head(10)
    pv=top["pct"].round(2).tolist()
    metric=(top["LS_GiaTriKhopLenh"]/1e9 if is_value else top["LS_KhoiLuongKhopLenh"])
    mlabel="GT (tỷ)" if is_value else "KL (CP)"
    cc=[["rgba(0,0,0,0)"]*10,["rgba(0,0,0,0)"]*10,[pct_clr(v) for v in pv],["rgba(0,0,0,0)"]*10,["rgba(0,0,0,0)"]*10]
    fig=go.Figure(go.Table(
        header=dict(values=["Mã","Giá","%","±",mlabel],fill_color="rgba(128,128,128,0.2)",align="left",
                    font=dict(size=13),height=36),
        cells=dict(values=[top["ticker"],top["LS_GiaDongCua_n"].round(0),
                           [f"{v:+.2f}%" for v in pv],top["chg"].round(0),
                           metric.map(lambda x:f"{x:,.1f}" if is_value else f"{x:,.0f}")],
                   fill_color=cc,align="left",font=dict(size=13),height=32)))
    title=f"Top 10 {'Giá trị' if is_value else 'Khối lượng'} GD — {D1.date()}"
    fig.update_layout(height=420,margin=dict(t=44,l=6,r=6,b=6),
        title=dict(text=title,font=dict(size=14),x=0))
    return fig

@st.cache_data(show_spinner=False)
def fig_net_chart(is_foreign):
    day=df[df["date"]==D1].copy()
    day["net"]=day["KN_GTDGRong"] if is_foreign else day["TD_GtMua"]-day["TD_GtBan"]
    buy=day[day["net"]>0].sort_values("net",ascending=False).head(10)
    sell=day[day["net"]<0].sort_values("net").head(10)
    bv=(buy["net"]/1e9).tolist(); sv=(sell["net"].abs()/1e9).tolist()
    amax=max(max(bv) if bv else 0,max(sv) if sv else 0)*1.35
    title="🌐 Khối ngoại" if is_foreign else "🏦 Tự doanh"
    fig=make_subplots(rows=1,cols=2,horizontal_spacing=.13,subplot_titles=("Bán ròng","Mua ròng"))
    fig.add_trace(go.Bar(x=sv,y=sell["ticker"].tolist(),orientation="h",marker_color="#f87171",
        text=[f"{v:.1f}T" for v in sv],textposition="outside",cliponaxis=False),1,1)
    fig.add_trace(go.Bar(x=bv,y=buy["ticker"].tolist(),orientation="h",marker_color="#4ade80",
        text=[f"{v:.1f}T" for v in bv],textposition="outside",cliponaxis=False),1,2)
    fig.update_xaxes(range=[amax,0],visible=False,row=1,col=1)
    fig.update_xaxes(range=[0,amax],visible=False,row=1,col=2)
    fig.update_yaxes(autorange="reversed",side="right",row=1,col=1)
    fig.update_yaxes(autorange="reversed",side="left", row=1,col=2)
    pass
    fig.update_layout(height=280,showlegend=False,
        margin=dict(t=80,l=50,r=50,b=10),
        title=dict(text=f"<b>{title} — Mua/Bán ròng</b><br><span style='font-size:12px;color:#475569'>Ngày {D1.date()}</span>",
                   x=.5,font=dict(size=16)))
    return fig

@st.cache_data(show_spinner=False)
def fig_rank(n_days, ascending):
    if len(ALL_DATES)<=n_days: return None
    d1,dp=ALL_DATES[-1],ALL_DATES[-(n_days+1)]
    now=df[df["date"]==d1][["ticker","LS_GiaDongCua"]]
    prv=df[df["date"]==dp][["ticker","LS_GiaDongCua"]]
    m=now.merge(prv,on="ticker",suffixes=("_n","_p"))
    m["pct"]=(m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"])/m["LS_GiaDongCua_p"]*100
    m["chg"]=m["LS_GiaDongCua_n"]-m["LS_GiaDongCua_p"]
    vol=df[(df["date"]>=dp)&(df["date"]<=d1)].groupby("ticker")["LS_KhoiLuongKhopLenh"].mean().reset_index()
    m=m.merge(vol,on="ticker",how="left")
    top=m.sort_values("pct",ascending=ascending).head(10)
    pv=top["pct"].round(2).tolist()
    cc=[["rgba(0,0,0,0)"]*10,["rgba(0,0,0,0)"]*10,[pct_clr(v) for v in pv],["rgba(0,0,0,0)"]*10,["rgba(0,0,0,0)"]*10]
    fig=go.Figure(go.Table(
        header=dict(values=["Mã","Giá","%","±","KL TB"],fill_color="rgba(128,128,128,0.2)",align="left",
                    font=dict(size=13),height=36),
        cells=dict(values=[top["ticker"],top["LS_GiaDongCua_n"].round(0),
                           [f"{v:+.2f}%" for v in pv],top["chg"].round(0),
                           top["LS_KhoiLuongKhopLenh"].map(lambda x:f"{x:,.0f}")],
                   fill_color=cc,align="left",font=dict(size=13),height=32)))
    fig.update_layout(height=410,margin=dict(t=10,l=6,r=6,b=6))
    return fig

@st.cache_data(show_spinner=False)
def calc_t3_money_flow_score(w1, w2, w3, spike_th, top_n):
    d = df.copy()
    if 'Volatility' not in d.columns:
        d['Volatility'] = d.groupby('ticker')['Daily_Return'].transform(lambda x: x.rolling(20, min_periods=1).std())
    if 'Volume_Ratio' not in d.columns:
        d['VMA_20'] = d.groupby('ticker')['LS_KhoiLuongKhopLenh'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        d['Volume_Ratio'] = d['LS_KhoiLuongKhopLenh'] / d['VMA_20']

    d_latest = d[d["date"] == D1].copy()
    d_latest['Daily_Return'] = d_latest['Daily_Return'].fillna(0)
    d_latest['Volume_Ratio'] = d_latest['Volume_Ratio'].fillna(1)
    d_latest['Volatility']   = d_latest['Volatility'].fillna(0)

    d_latest['score'] = (w1 * d_latest['Daily_Return']) + \
                        (w2 * d_latest['Volume_Ratio']) - \
                        (w3 * d_latest['Volatility'])
    d_latest = d_latest.sort_values("score", ascending=False).reset_index(drop=True)

    # Khuyến nghị Mua / Bán / Quan sát
    BUY_SCORE_THRESHOLD = 0.5
    SELL_RETURN_THRESHOLD = -1.0
    
    conditions = [
        (d_latest['score'] > BUY_SCORE_THRESHOLD) & (d_latest['Volume_Ratio'] >= spike_th) & (d_latest['Daily_Return'] > 0),
        (d_latest['Daily_Return'] < SELL_RETURN_THRESHOLD) & (d_latest['Volume_Ratio'] > spike_th)
    ]
    choices = ['BUY', 'SELL / AVOID']
    d_latest['recommendation'] = np.select(conditions, choices, default='OBSERVE')

    # Đột biến khối lượng (Spikes)
    df_spikes = d_latest[d_latest['Volume_Ratio'] >= spike_th].sort_values(by='Volume_Ratio', ascending=True).tail(10)
    colors_spk = ["#3b82f6"] * len(df_spikes)
    fig_spike = go.Figure(go.Bar(x=df_spikes["Volume_Ratio"], y=df_spikes["ticker"], orientation="h",
                                 marker=dict(color=colors_spk, line_width=0),
                                 text=[f"{v:.1f}x" for v in df_spikes["Volume_Ratio"]], textposition="outside"))
    fig_spike.update_layout(height=260, title=dict(text=f"🔥 Top Đột Biến KL (Ratio ≥ {spike_th})", font=dict(size=14), x=0),
                            **dl(margin=dict(t=50,l=10,r=40,b=10)), xaxis_title="Volume Ratio")

    # Flow Score (T+3)
    top_score = d_latest.head(top_n).sort_values("score", ascending=True)
    colors_score = [score_clr(v * 2) for v in top_score["score"]]
    fig_score = go.Figure(go.Bar(x=top_score["score"], y=top_score["ticker"], orientation="h",
                                 marker=dict(color=colors_score, line_width=0),
                                 text=[f"{v:.2f}" for v in top_score["score"]], textposition="outside"))
    fig_score.update_layout(height=260, title=dict(text="💹 Flow Score (T+3) — Dòng tiền thông minh", font=dict(size=14), x=0),
                            **dl(margin=dict(t=50,l=10,r=40,b=10)), xaxis_title="Điểm T+3")

    table_cols = ["ticker", "score", "Daily_Return", "Volume_Ratio", "Volatility", "recommendation"]
    return fig_spike, fig_score, d_latest[table_cols]

def make_stock_chart(ticker, n_days):
    ds=df[df["ticker"]==ticker].tail(n_days)
    if ds.empty: return None,None,None
    # Candle + Volume
    fig1=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.72,.28],vertical_spacing=.03)
    fig1.add_trace(go.Candlestick(x=ds["date"],open=ds["LS_GiaMoCua"],high=ds["LS_GiaCaoNhat"],
        low=ds["LS_GiaThapNhat"],close=ds["LS_GiaDongCua"],name="Giá",
        increasing_line_color="#4ade80",decreasing_line_color="#f87171",
        increasing_fillcolor="#4ade80",decreasing_fillcolor="#f87171"),1,1)
    for col,clr,nm in [("SMA_20","#fbbf24","SMA20"),("SMA_50","#60a5fa","SMA50"),("SMA_200","#c084fc","SMA200")]:
        if col in ds.columns:
            fig1.add_trace(go.Scatter(x=ds["date"],y=ds[col],line=dict(color=clr,width=1.2),name=nm,opacity=.9),1,1)
    vc=["#4ade80" if c>=o else "#f87171" for c,o in zip(ds["LS_GiaDongCua"],ds["LS_GiaMoCua"])]
    fig1.add_trace(go.Bar(x=ds["date"],y=ds["LS_KhoiLuongKhopLenh"],name="KL",marker_color=vc,opacity=.65),2,1)
    if "VMA_20" in ds.columns:
        fig1.add_trace(go.Scatter(x=ds["date"],y=ds["VMA_20"],line=dict(color="#fbbf24",width=1.2),name="VMA20"),2,1)
    fig1.update_layout(height=520,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.05,x=0),
        title=dict(text=f"📈 {ticker} — Biểu đồ nến",font=dict(size=16),x=0),
        **dl(margin=dict(t=60,l=10,r=10,b=5)))
    fig1.update_yaxes(row=1,col=1)
    fig1.update_yaxes(row=2,col=1)
    fix_date_gaps(fig1)
    # RSI
    fig2=None
    if "RSI" in ds.columns:
        d2=ds.dropna(subset=["RSI"])
        fig2=go.Figure()
        fig2.add_hrect(y0=70,y1=100,fillcolor="rgba(248,113,113,.07)",line_width=0)
        fig2.add_hrect(y0=0, y1=30,fillcolor="rgba(74,222,128,.07)",line_width=0)
        fig2.add_hline(y=70,line=dict(color="#f87171",width=1,dash="dot"),
            annotation_text="Quá mua (70)",annotation_font_color="#f87171",annotation_position="bottom right")
        fig2.add_hline(y=30,line=dict(color="#4ade80",width=1,dash="dot"),
            annotation_text="Quá bán (30)",annotation_font_color="#4ade80",annotation_position="top right")
        fig2.add_trace(go.Scatter(x=d2["date"],y=d2["RSI"],line=dict(color="#60a5fa",width=1.8),name="RSI"))
        fig2.update_layout(height=250,yaxis_range=[0,100],
            title=dict(text="RSI (14)",font=dict(size=14),x=0),
            **dl(margin=dict(t=40,l=10,r=10,b=5)))
        fix_date_gaps(fig2)
    # MACD
    fig3=None
    cols=["MACD","Signal_Line","MACD_Histogram"]
    if all(c in ds.columns for c in cols):
        d3=ds.dropna(subset=cols)
        ch=["#4ade80" if v>=0 else "#f87171" for v in d3["MACD_Histogram"]]
        fig3=go.Figure()
        fig3.add_trace(go.Bar(x=d3["date"],y=d3["MACD_Histogram"],name="Histogram",marker_color=ch,opacity=.75))
        fig3.add_trace(go.Scatter(x=d3["date"],y=d3["MACD"],line=dict(color="#60a5fa",width=1.5),name="MACD"))
        fig3.add_trace(go.Scatter(x=d3["date"],y=d3["Signal_Line"],line=dict(color="#f97316",width=1.5),name="Signal"))
        fig3.update_layout(height=280,
            title=dict(text="MACD (12,26,9)",font=dict(size=14),x=0),
            legend=dict(orientation="h",y=1.1,xanchor='right',x=1),
            **dl(margin=dict(t=60,l=10,r=10,b=5)))
        fix_date_gaps(fig3)
    return fig1,fig2,fig3

# ════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:10px 0 6px'>
      <div style='font-size:1.7rem'>📈</div>
      <div style='font-size:1.1rem;font-weight:800;color:var(--text-color)'>VN100 Dashboard</div>
      <div style='font-size:.72rem;color:var(--text-color);opacity:0.7;margin-top:2px'>{pd.to_datetime(D1).date()} • {len(TICKERS)} cổ phiếu</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    PAGE = st.radio("", [
        "🏠  Thị trường",
        "📊  Ngành & Giao dịch",
        "💸  Dòng tiền",
        "📈  Phân tích CP",
        "🤖  Dự đoán T+",
    ], label_visibility="collapsed")

    st.markdown("---")

    # ── Dynamic controls per page ────────────────────────────────
    if PAGE == "🏠  Thị trường":
        st.markdown("<div class='ctrl-title'>🗺️ Treemap</div>", unsafe_allow_html=True)
        top_n_tree = st.slider("Số ngành hiển thị", 5, 20, 10, key="tree_n")

    elif PAGE == "📊  Ngành & Giao dịch":
        st.markdown("<div class='ctrl-title'>📊 Hiệu suất ngành</div>", unsafe_allow_html=True)
        sec_period = st.radio("Kỳ thời gian", ["1 Ngày","1 Tuần","1 Tháng"],
                              horizontal=False, key="sec_kp")
        sec_days = {"1 Ngày":1,"1 Tuần":7,"1 Tháng":30}[sec_period]

    elif PAGE == "💸  Dòng tiền":
        pass # All controls moved to main page

    elif PAGE == "📈  Phân tích CP":
        st.markdown("<div class='ctrl-title'>🔍 Chọn cổ phiếu</div>", unsafe_allow_html=True)
        sel_ticker = st.selectbox("Mã CP", TICKERS,
                                  index=TICKERS.index("VCB") if "VCB" in TICKERS else 0,
                                  key="sel_tk")
        st.markdown("<div class='ctrl-title' style='margin-top:14px'>📅 Lịch sử</div>", unsafe_allow_html=True)
        n_days_map = {"3 tháng":60,"6 tháng":120,"1 năm":250,"2 năm":500,"Toàn bộ":9999}
        n_days_lbl = st.radio("Khoảng thời gian",list(n_days_map.keys()),index=1,key="nd")
        n_days = n_days_map[n_days_lbl]
        st.markdown("<div class='ctrl-title' style='margin-top:14px'>📉 Chỉ báo</div>", unsafe_allow_html=True)
        show_sma = st.checkbox("SMA (20/50/200)", value=True)
        show_rsi  = st.checkbox("RSI (14)", value=True)
        show_macd = st.checkbox("MACD", value=True)

    elif PAGE == "🤖  Dự đoán T+":
        st.markdown("<div class='ctrl-title'>🤖 Thiết lập mô hình</div>", unsafe_allow_html=True)
        pred_symbols = st.multiselect("Mã Cổ phiếu (Trống = Top 10)", TICKERS, key="pred_sym")
        pred_months = st.slider("Kỳ quan sát (tháng)", 1, 36, 24, key="pred_m")
        run_pred = st.button("🚀 Chạy Dự Đoán", use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='color:var(--text-color);opacity:0.6;font-size:.7rem;text-align:center'>VN100 Dashboard © 2026</div>",
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# KPI BAR (always visible)
# ════════════════════════════════════════════════════════════════════
kpi = kpi_data()
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:12px'>
    <h1 style='background:linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;font-size:1.6rem;margin:0'>📊 VN100 Dashboard</h1>
    <div style='color:var(--text-color);opacity:0.8;font-size:.9rem;font-weight:500'>
    ⏱️ Phiên: <span style='color:#3b82f6;font-weight:700'>{D1.strftime('%d/%m/%Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📅 Phiên GD",    str(D1.date()))
c2.metric("💰 GTGD",        f"{kpi['val']:.2f} ngàn tỷ")
c3.metric("📈 Tăng",        f"{kpi['up']}",  f"Giảm {kpi['dn']} | TC {kpi['ref']}")
c4.metric("🌐 Khối ngoại",  f"{kpi['nn']:+.1f} tỷ")
c5.metric("📦 KL",          f"{kpi['vol']:.0f}M CP")
c6.metric("🏢 Số CP",       f"{kpi['nstk']}")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════
# PAGE CONTENT
# ════════════════════════════════════════════════════════════════════

# ── PAGE 1: Thị trường ───────────────────────────────────────────
if PAGE == "🏠  Thị trường":
    st.plotly_chart(fig_treemap(top_n_tree), use_container_width=True)

    c1,c2 = st.columns([3,2])
    with c1:
        st.plotly_chart(fig_breadth(),      use_container_width=True)
    with c2:
        st.plotly_chart(fig_vol_trend(),    use_container_width=True)

    st.markdown("<div class='sh'>🌐 Giao dịch Khối ngoại & Tự doanh</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_net_chart(True),  use_container_width=True)
    st.plotly_chart(fig_net_chart(False), use_container_width=True)

    st.markdown("<div class='sh'>🏆 Xếp hạng cổ phiếu (1 Tuần)</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📈 Top tăng giá")
        f = fig_rank(7, ascending=False)
        if f: st.plotly_chart(f, use_container_width=True)
    with col2:
        st.markdown("##### 📉 Top giảm giá")
        f = fig_rank(7, ascending=True)
        if f: st.plotly_chart(f, use_container_width=True)

    st.markdown("<div class='sh'>📊 Top giao dịch phiên gần nhất</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig_top_table("LS_KhoiLuongKhopLenh",False),use_container_width=True)
    with col4: st.plotly_chart(fig_top_table("LS_GiaTriKhopLenh",True), use_container_width=True)

# ── PAGE 2: Ngành & Giao dịch ───────────────────────────────────
elif PAGE == "📊  Ngành & Giao dịch":
    t1, t2 = st.tabs(["📋 Top giao dịch","📊 Hiệu suất ngành"])

    with t1:
        col1,col2 = st.columns(2)
        with col1: st.plotly_chart(fig_top_table("LS_KhoiLuongKhopLenh",False),use_container_width=True)
        with col2: st.plotly_chart(fig_top_table("LS_GiaTriKhopLenh",True), use_container_width=True)

    with t2:
        fig_sb = fig_sector_bar(sec_days)
        if fig_sb: st.plotly_chart(fig_sb, use_container_width=True)

# ── PAGE 3: Dòng tiền ───────────────────────────────────────────
elif PAGE == "💸  Dòng tiền":
    st.markdown("### ⚙️ Tuỳ chỉnh mô hình Flow Score (T+3)")
    st.info("Kéo các thanh trượt bên dưới để thay đổi tỷ trọng các bộ lọc dòng tiền thông minh theo mong muốn của bạn.")
    
    # Sliders moved here for less confusing sidebar
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.w_ret = c1.slider("🎯 Lợi nhuận (W1)", 0.0, 1.0, float(st.session_state.w_ret), 0.1)
    st.session_state.w_vol = c2.slider("📦 Khối lượng (W2)", 0.0, 1.0, float(st.session_state.w_vol), 0.1)
    st.session_state.w_volat = c3.slider("⚖️ Rủi ro (W3)", 0.0, 1.0, float(st.session_state.w_volat), 0.1)
    st.session_state.vol_spk = c4.slider("🔥 Ngưỡng nổ KL", 1.0, 5.0, float(st.session_state.vol_spk), 0.1)
    
    w_ret, w_vol, w_volat, vol_spike_th = st.session_state.w_ret, st.session_state.w_vol, st.session_state.w_volat, st.session_state.vol_spk
    top_n_flow = 10  # Mặc định top 10

    st.markdown("---")

    fig_spike, fig_score, df_fl = calc_t3_money_flow_score(w_ret, w_vol, w_volat, vol_spike_th, top_n_flow)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_spike, use_container_width=True)
    with col2:
        st.plotly_chart(fig_score, use_container_width=True)

    with st.expander("📋 Bảng chi tiết Flow Score (T+3) Xếp hạng", expanded=True):
        df_display = df_fl.rename(columns={
            "ticker":"Mã CP", "score":"T+3 Score", "Daily_Return":"Return (%)", 
            "Volume_Ratio":"Volume Ratio (x)", "Volatility":"Độ rủi ro", "recommendation":"Khuyến nghị"
        })
        num_cols = ["T+3 Score", "Return (%)", "Volume Ratio (x)", "Độ rủi ro"]
        df_display[num_cols] = df_display[num_cols].round(2)
        st.dataframe(df_display, use_container_width=True, height=250)



# ── PAGE 5: Phân tích cổ phiếu ──────────────────────────────────
elif PAGE == "📈  Phân tích CP":
    ds = df[df["ticker"]==sel_ticker].copy()
    if n_days < 9999: ds = ds.tail(n_days)
    last = ds.iloc[-1]; prev = ds.iloc[-2] if len(ds)>1 else last
    chg = last["LS_GiaDongCua"]-prev["LS_GiaDongCua"]
    pct = chg/prev["LS_GiaDongCua"]*100
    rsi_val = last.get("RSI",None)

    # Mini KPI row for stock
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("📌 Mã CP",    sel_ticker)
    k2.metric("💵 Giá đóng", f"{last['LS_GiaDongCua']:,.0f}", f"{pct:+.2f}%")
    k3.metric("🔺 Cao nhất", f"{last['LS_GiaCaoNhat']:,.0f}")
    k4.metric("🔻 Thấp nhất",f"{last['LS_GiaThapNhat']:,.0f}")
    k5.metric("📦 KL",       f"{last['LS_KhoiLuongKhopLenh']:,.0f}")
    if rsi_val and not pd.isna(rsi_val):
        rsi_state="Quá mua" if rsi_val>70 else ("Quá bán" if rsi_val<30 else "Trung lập")
        k6.metric("📉 RSI",  f"{rsi_val:.1f}", rsi_state)
    st.markdown("---")

    # Candlestick (SMA toggle applied via filter on ds)
    if not show_sma:
        for c in ["SMA_20","SMA_50","SMA_200"]: ds[c] = np.nan
    fc, fr, fm = make_stock_chart(sel_ticker, n_days if n_days<9999 else len(ds))
    if fc: st.plotly_chart(fc, use_container_width=True)

    if show_rsi and show_macd:
        r1,r2 = st.columns(2)
        with r1:
            if fr: st.plotly_chart(fr, use_container_width=True)
        with r2:
            if fm: st.plotly_chart(fm, use_container_width=True)
    elif show_rsi:
        if fr: st.plotly_chart(fr, use_container_width=True)
    elif show_macd:
        if fm: st.plotly_chart(fm, use_container_width=True)

    with st.expander("📋 Xem dữ liệu thô"):
        cols_show=["date","LS_GiaMoCua","LS_GiaCaoNhat","LS_GiaThapNhat",
                   "LS_GiaDongCua","LS_KhoiLuongKhopLenh","Daily_Return","RSI","MACD"]
        cols_show=[c for c in cols_show if c in ds.columns]
        st.dataframe(ds[cols_show].sort_values("date",ascending=False).reset_index(drop=True),
                     use_container_width=True, height=280)

# ── PAGE 6: Dự đoán T+ (ML) ─────────────────────────────────────
elif PAGE == "🤖  Dự đoán T+":
    st.markdown("### 📋 Dự đoán xu hướng T+")
    st.markdown("<div style='color:var(--text-color);opacity:0.8;margin-bottom:20px'>Hệ thống sử dụng các mô hình LightGBM được huấn luyện sẵn để dự đoán xu hướng T+3, T+7, T+15, và T+30.</div>", unsafe_allow_html=True)

    # Hàm chạy ẩn kết hợp cache bộ nhớ để ko gọi đi gọi lại
    @st.cache_data(show_spinner=False)
    def run_ml_pipeline(sym_tuple, months):
        m_csv = test_model_dir.parent / "dataset.csv"
        m_dir = test_model_dir.parent / "models"
        return predict_symbols_from_master_csv(
            master_csv_path=m_csv,
            models_dir=m_dir,
            output_dir=None,
            mode="recent_months",
            recent_months=months,
            symbols=list(sym_tuple) if sym_tuple else None,
            default_top_symbols=10
        )

    if run_pred:
        with st.spinner("⏳ Đang tải dữ liệu và chạy Inference (Vui lòng chờ)..."):
            try:
                res = run_ml_pipeline(tuple(pred_symbols), pred_months)
                
                # Hiển thị bảng
                st.markdown("##### 🏆 Bảng Dự Đoán Mới Nhất")
                st.dataframe(res["latest_selected_symbol_table"].style.format(precision=3), use_container_width=True)
                


                st.markdown("---")
                st.markdown("##### 📊 So sánh 4 Horizon")
                st.markdown("> *Ghi chú: Thanh màu xanh là xác suất nghiêng về TĂNG (>0.5), thanh màu đỏ là nghiêng về GIẢM (<0.5).*")
                fig_latest = plot_latest_symbol_signal(res)
                st.pyplot(fig_latest, use_container_width=True)
                
                # Hỗ trợ giải phóng RAM ngay lập tức
                import gc
                gc.collect()

            except Exception as e:
                st.error(f"❌ Xảy ra lỗi khi chạy mô hình: {e}")

