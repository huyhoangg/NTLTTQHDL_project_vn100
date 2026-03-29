import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="VN100 Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:#0c111d}
section[data-testid="stSidebar"]{background:#0f1623;border-right:1px solid #1e293b}
/* Nav radio styling */
div[data-testid="stSidebar"] .stRadio>div{gap:4px}
div[data-testid="stSidebar"] .stRadio label{
    display:block;width:100%;padding:10px 14px;border-radius:10px;
    cursor:pointer;color:#94a3b8;font-weight:500;font-size:.92rem;
    transition:all .15s;border:1px solid transparent;
}
div[data-testid="stSidebar"] .stRadio label:hover{background:#1e293b;color:#e2e8f0}
div[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
div[data-testid="stSidebar"] .stRadio input:checked + div label{
    background:linear-gradient(135deg,#1d4ed8,#6d28d9);color:#fff;border-color:#3b82f6
}
/* Metric cards */
div[data-testid="metric-container"]{
    background:linear-gradient(135deg,#131f35,#1a2540);
    border:1px solid #1e3a5f;border-radius:12px;padding:14px 18px;
    box-shadow:0 4px 20px rgba(0,0,0,.5);transition:transform .18s,box-shadow .18s
}
div[data-testid="metric-container"]:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(37,99,235,.2)}
div[data-testid="metric-container"] label{color:#7dd3fc!important;font-size:.76rem!important;font-weight:600!important}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#f1f5f9!important;font-weight:700!important}
/* Control card in sidebar */
.ctrl-card{background:#131f35;border:1px solid #1e293b;border-radius:12px;padding:14px;margin-bottom:12px}
.ctrl-title{color:#60a5fa;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
/* Section header */
.sh{background:linear-gradient(90deg,rgba(37,99,235,.2),transparent);
    border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;
    padding:8px 14px;margin:16px 0 10px;color:#93c5fd;font-weight:600;font-size:1rem}
/* Selectbox, slider */
.stSelectbox>div>div,.stMultiSelect>div>div{
    background:#131f35!important;border:1px solid #1e3a5f!important;
    border-radius:8px!important;color:#e2e8f0!important}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:#3b82f6!important}
hr{border-color:#1e293b!important;margin:.8rem 0!important}
h1,h2,h3{color:#e2e8f0!important}
/* Tabs (secondary within page) */
.stTabs [data-baseweb="tab-list"]{background:#131f35;border-radius:10px;padding:4px;gap:3px;border:1px solid #1e293b}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#64748b;font-size:.85rem;padding:6px 14px}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1d4ed8,#6d28d9)!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BG = "rgba(0,0,0,0)"; GR = "rgba(255,255,255,.04)"; AX = "#475569"; TX = "#94a3b8"

def pct_clr(v): return "#34d399" if v>0 else ("#f87171" if v<0 else "#fbbf24")
def score_clr(s):
    if pd.isna(s): return "#6b7280"
    if s<=-7: return "#22d3ee"
    if s<-3:  return "#f87171"
    if s<0:   return "#fca5a5"
    if s==0:  return "#fbbf24"
    if s<3:   return "#4ade80"
    if s<7:   return "#16a34a"
    return "#a855f7"

def dl(**kw):
    """Dark layout preset"""
    return dict(paper_bgcolor=BG,plot_bgcolor=BG,
                font=dict(color=TX,family="Inter"),
                xaxis=dict(gridcolor=GR,color=AX,zerolinecolor=GR),
                yaxis=dict(gridcolor=GR,color=AX,zerolinecolor=GR),**kw)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Đang tải dữ liệu...")
def load_data():
    df = pd.read_csv("vn100_featured_data.csv", parse_dates=["date"])
    return df.sort_values(["ticker","date"]).reset_index(drop=True)

df = load_data()
ALL_DATES = sorted(df["date"].unique())
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
        marker=dict(colors=ind["color"],line=dict(color="#0c111d",width=2)),
        hovertemplate="<b>%{label}</b><br>GTGD: %{value:,.0f} tỷ<extra></extra>",
        tiling=dict(pad=3,packing="squarify"),
    ))
    fig.update_layout(height=500,paper_bgcolor=BG,margin=dict(t=40,l=4,r=4,b=4),
        title=dict(text=f"🗺️ Bản đồ ngành — {D1.date()}",font=dict(size=18,color="#e2e8f0"),x=.5))
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
    fig.update_layout(barmode="relative",height=300,legend=dict(bgcolor=BG,font_color=TX,orientation="h",y=1.1),
        title=dict(text="📊 Độ rộng thị trường",font=dict(size=15,color="#e2e8f0"),x=0),
        **dl(margin=dict(t=50,l=10,r=10,b=10)))
    return fig

@st.cache_data(show_spinner=False)
def fig_vol_trend():
    rows=[{"date":d,"val":df[df["date"]==d]["LS_GiaTriKhopLenh"].sum()/1e12}
          for d in ALL_DATES[-60:]]
    vt=pd.DataFrame(rows)
    ma=vt["val"].rolling(5).mean()
    fig=go.Figure()
    fig.add_trace(go.Bar(x=vt["date"],y=vt["val"],name="GTGD (nghìn tỷ)",marker_color="#3b82f6",opacity=.7))
    fig.add_trace(go.Scatter(x=vt["date"],y=ma,name="MA5",line=dict(color="#fbbf24",width=2)))
    fig.update_layout(height=270,legend=dict(bgcolor=BG,font_color=TX),
        title=dict(text="💰 Giá trị GD toàn thị trường",font=dict(size=15,color="#e2e8f0"),x=0),
        **dl(margin=dict(t=50,l=10,r=10,b=10)),yaxis_title="Nghìn tỷ")
    return fig

@st.cache_data(show_spinner=False)
def fig_sector_heatmap():
    recent=ALL_DATES[-20:]
    rows=[]
    for d in recent:
        day=df[df["date"]==d]
        rows.append(day.groupby("icb_name2")["Daily_Return"].mean())
    heat=pd.DataFrame(rows,index=[d.strftime("%m/%d") for d in recent]).dropna(axis=1,thresh=8)
    fig=go.Figure(go.Heatmap(z=heat.values,x=heat.columns.tolist(),y=heat.index.tolist(),
        colorscale=[[0,"#ef4444"],[.5,"#111827"],[1,"#22c55e"]],zmid=0,zmin=-3,zmax=3,
        colorbar=dict(title="Return%",tickfont_color=TX,title_font_color=TX),
        text=np.round(heat.values,1),texttemplate="%{text:.1f}",textfont_size=9))
    fig.update_layout(height=460,paper_bgcolor=BG,plot_bgcolor=BG,
        xaxis=dict(color=AX,tickfont_size=10),yaxis=dict(color=AX,tickfont_size=10),
        title=dict(text="🌡️ Heatmap lợi suất ngành (20 phiên)",font=dict(size=15,color="#e2e8f0"),x=0),
        margin=dict(t=50,l=10,r=10,b=100))
    return fig

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
        text=[f"{v:+.2f}%" for v in sec["pct"]],textposition="outside",textfont_color="#e2e8f0"))
    fig.update_layout(height=420,title=dict(text=f"📊 Hiệu suất ngành — {label}",font=dict(size=15,color="#e2e8f0"),x=0),
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
    cc=[["#0c1a2e"]*10,["#0c1a2e"]*10,[pct_clr(v) for v in pv],["#0c1a2e"]*10,["#0c1a2e"]*10]
    fig=go.Figure(go.Table(
        header=dict(values=["Mã","Giá","%","±",mlabel],fill_color="#1e3a5f",align="left",
                    font=dict(color="#7dd3fc",size=13),height=36),
        cells=dict(values=[top["ticker"],top["LS_GiaDongCua_n"].round(0),
                           [f"{v:+.2f}%" for v in pv],top["chg"].round(0),
                           metric.map(lambda x:f"{x:,.1f}" if is_value else f"{x:,.0f}")],
                   fill_color=cc,align="left",font=dict(color="#e2e8f0",size=13),height=32)))
    title=f"Top 10 {'Giá trị' if is_value else 'Khối lượng'} GD — {D1.date()}"
    fig.update_layout(height=420,paper_bgcolor=BG,margin=dict(t=44,l=6,r=6,b=6),
        title=dict(text=title,font=dict(size=14,color="#93c5fd"),x=0))
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
    fig.update_yaxes(autorange="reversed",side="right",tickfont_color=TX,row=1,col=1)
    fig.update_yaxes(autorange="reversed",side="left", tickfont_color=TX,row=1,col=2)
    for a in fig.layout.annotations: a.font.color="#94a3b8"
    fig.update_layout(height=440,showlegend=False,paper_bgcolor=BG,plot_bgcolor=BG,
        margin=dict(t=80,l=50,r=50,b=10),
        title=dict(text=f"<b>{title} — Mua/Bán ròng</b><br><span style='font-size:12px;color:#475569'>Ngày {D1.date()}</span>",
                   x=.5,font=dict(size=16,color="#e2e8f0")))
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
    cc=[["#0c1a2e"]*10,["#0c1a2e"]*10,[pct_clr(v) for v in pv],["#0c1a2e"]*10,["#0c1a2e"]*10]
    fig=go.Figure(go.Table(
        header=dict(values=["Mã","Giá","%","±","KL TB"],fill_color="#1e3a5f",align="left",
                    font=dict(color="#7dd3fc",size=13),height=36),
        cells=dict(values=[top["ticker"],top["LS_GiaDongCua_n"].round(0),
                           [f"{v:+.2f}%" for v in pv],top["chg"].round(0),
                           top["LS_KhoiLuongKhopLenh"].map(lambda x:f"{x:,.0f}")],
                   fill_color=cc,align="left",font=dict(color="#e2e8f0",size=13),height=32)))
    fig.update_layout(height=410,paper_bgcolor=BG,margin=dict(t=10,l=6,r=6,b=6))
    return fig

@st.cache_data(show_spinner=False)
def calc_flow(top_n):
    d=df.copy().sort_values(["ticker","date"])
    g=d.groupby("ticker",sort=False)
    d["ret5"]=g["LS_GiaDongCua"].pct_change(5)*100
    d["typ"]=(d["LS_GiaCaoNhat"]+d["LS_GiaThapNhat"]+d["LS_GiaDongCua"])/3
    d["rmf"]=d["typ"]*d["LS_KhoiLuongKhopLenh"]
    tpp=g["typ"].shift(1)
    d["pmf"]=np.where(d["typ"]>tpp,d["rmf"],0.)
    d["nmf"]=np.where(d["typ"]<tpp,d["rmf"],0.)
    d["p14"]=g["pmf"].rolling(14,min_periods=14).sum().reset_index(level=0,drop=True)
    d["n14"]=g["nmf"].rolling(14,min_periods=14).sum().reset_index(level=0,drop=True)
    d["mfi"]=100-(100/(1+d["p14"]/d["n14"].replace(0,np.nan)))
    sp=(d["LS_GiaCaoNhat"]-d["LS_GiaThapNhat"]).replace(0,np.nan)
    d["mfm"]=((d["LS_GiaDongCua"]-d["LS_GiaThapNhat"])-(d["LS_GiaCaoNhat"]-d["LS_GiaDongCua"]))/sp
    d["mfm"]=d["mfm"].replace([np.inf,-np.inf],np.nan).fillna(0).clip(-1,1)
    d["mfv"]=d["mfm"]*d["LS_KhoiLuongKhopLenh"]
    d["cmf"]=(g["mfv"].rolling(21,min_periods=21).sum()/g["LS_KhoiLuongKhopLenh"].rolling(21,min_periods=21).sum()).reset_index(level=0,drop=True)
    lat=g.tail(1).copy().dropna(subset=["ret5","mfi","cmf"])
    lat["score"]=100*(0.5*lat["cmf"].clip(-1,1)+0.3*(lat["mfi"]-50)/50+0.2*lat["ret5"].clip(-10,10)/10)
    lat=lat.sort_values("score",ascending=False).reset_index(drop=True)
    top=lat.head(top_n).sort_values("score",ascending=True)
    colors=[score_clr(v/7) for v in top["score"]]
    fig=go.Figure(go.Bar(x=top["score"],y=top["ticker"],orientation="h",
        marker=dict(color=colors,line_width=0),
        text=[f"{v:.1f}" for v in top["score"]],textposition="outside",textfont_color="#e2e8f0"))
    fig.update_layout(height=380,title=dict(text="💹 Flow Score — Top dòng tiền mạnh",
        font=dict(size=15,color="#e2e8f0"),x=0),
        **dl(margin=dict(t=50,l=10,r=60,b=10)),xaxis_title="Điểm")
    return fig, lat[["ticker","score","mfi","cmf","ret5"]].head(30).round(2)

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
        legend=dict(bgcolor=BG,font_color=TX,orientation="h",y=1.05,x=0),
        title=dict(text=f"📈 {ticker} — Biểu đồ nến",font=dict(size=16,color="#e2e8f0"),x=0),
        **dl(margin=dict(t=60,l=10,r=10,b=5)))
    fig1.update_yaxes(gridcolor=GR,color=AX,row=1,col=1)
    fig1.update_yaxes(gridcolor=GR,color=AX,row=2,col=1)
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
            title=dict(text="RSI (14)",font=dict(size=14,color="#94a3b8"),x=0),
            **dl(margin=dict(t=40,l=10,r=10,b=5)))
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
        fig3.update_layout(height=250,
            title=dict(text="MACD (12,26,9)",font=dict(size=14,color="#94a3b8"),x=0),
            legend=dict(bgcolor=BG,font_color=TX,orientation="h",y=1.2),
            **dl(margin=dict(t=50,l=10,r=10,b=5)))
    return fig1,fig2,fig3

# ════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:10px 0 6px'>
      <div style='font-size:1.7rem'>📈</div>
      <div style='font-size:1.1rem;font-weight:800;color:#f1f5f9'>VN100 Dashboard</div>
      <div style='font-size:.72rem;color:#475569;margin-top:2px'>{D1.date()} • {len(TICKERS)} cổ phiếu</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    PAGE = st.radio("", [
        "🏠  Thị trường",
        "📊  Ngành & Giao dịch",
        "💸  Dòng tiền",
        "🏆  Xếp hạng",
        "📈  Phân tích CP",
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
        st.markdown("<div class='ctrl-title'>💹 Flow Score</div>", unsafe_allow_html=True)
        top_n_flow = st.slider("Số cổ phiếu", 5, 25, 10, key="flow_n")
        st.markdown("<div class='ctrl-title' style='margin-top:14px'>👁️ Hiển thị</div>", unsafe_allow_html=True)
        show_nn = st.checkbox("Khối ngoại", value=True)
        show_td = st.checkbox("Tự doanh",   value=True)

    elif PAGE == "🏆  Xếp hạng":
        st.markdown("<div class='ctrl-title'>📅 Kỳ xếp hạng</div>", unsafe_allow_html=True)
        rk_label = st.radio("Chọn kỳ", ["1 Ngày","1 Tuần","1 Tháng"],
                            horizontal=False, key="rk_p")
        rk_days = {"1 Ngày":1,"1 Tuần":7,"1 Tháng":30}[rk_label]

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

    st.markdown("---")
    st.markdown("<div style='color:#1e293b;font-size:.7rem;text-align:center'>VN100 Dashboard © 2026</div>",
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# KPI BAR (always visible)
# ════════════════════════════════════════════════════════════════════
kpi = kpi_data()
st.markdown(f"""
<h1 style='background:linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
background-clip:text;font-size:1.9rem;margin:0 0 2px'>📊 VN100 Market Dashboard</h1>
<p style='color:#334155;font-size:.85rem;margin-top:0'>Dữ liệu phiên {D1.date()} — Tương tác đầy đủ</p>
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

# ── PAGE 2: Ngành & Giao dịch ───────────────────────────────────
elif PAGE == "📊  Ngành & Giao dịch":
    t1, t2, t3 = st.tabs(["📋 Top giao dịch","📊 Hiệu suất ngành","🌡️ Heatmap ngành"])

    with t1:
        col1,col2 = st.columns(2)
        with col1: st.plotly_chart(fig_top_table("LS_KhoiLuongKhopLenh",False),use_container_width=True)
        with col2: st.plotly_chart(fig_top_table("LS_GiaTriKhopLenh",True), use_container_width=True)

    with t2:
        fig_sb = fig_sector_bar(sec_days)
        if fig_sb: st.plotly_chart(fig_sb, use_container_width=True)

    with t3:
        st.plotly_chart(fig_sector_heatmap(), use_container_width=True)

# ── PAGE 3: Dòng tiền ───────────────────────────────────────────
elif PAGE == "💸  Dòng tiền":
    if show_nn and "KN_GTDGRong" in df.columns:
        st.plotly_chart(fig_net_chart(True),  use_container_width=True)
        st.markdown("---")
    if show_td and "TD_GtMua" in df.columns:
        st.plotly_chart(fig_net_chart(False), use_container_width=True)
        st.markdown("---")

    fig_fl, df_fl = calc_flow(top_n_flow)
    st.plotly_chart(fig_fl, use_container_width=True)
    with st.expander("📋 Bảng chi tiết Flow Score"):
        st.dataframe(df_fl.rename(columns={"mfi":"MFI14","cmf":"CMF21","ret5":"Return 5D%"}),
                     use_container_width=True, height=300)

# ── PAGE 4: Xếp hạng ────────────────────────────────────────────
elif PAGE == "🏆  Xếp hạng":
    st.markdown(f"<div class='sh'>Xếp hạng kỳ: <b>{rk_label}</b></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📈 Top tăng giá")
        f = fig_rank(rk_days, ascending=False)
        if f: st.plotly_chart(f, use_container_width=True)
    with col2:
        st.markdown("##### 📉 Top giảm giá")
        f = fig_rank(rk_days, ascending=True)
        if f: st.plotly_chart(f, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='sh'>Giao dịch phiên gần nhất</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig_top_table("LS_KhoiLuongKhopLenh",False),use_container_width=True)
    with col4: st.plotly_chart(fig_top_table("LS_GiaTriKhopLenh",True), use_container_width=True)

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
