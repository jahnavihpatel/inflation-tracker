"""
Inflation Tracker v2 — Streamlit Dashboard
Now with: News Sentiment · Smart Shopping Advisor · Geopolitical Risk Index
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="US Inflation Tracker",page_icon="📈",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.metric-card{background:#13161e;border:1px solid #1e2330;border-radius:12px;padding:1.1rem 1.4rem;}
.metric-label{font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#5a6070;margin-bottom:6px;}
.metric-value{font-size:24px;font-weight:600;color:#e8eaf0;font-family:'DM Mono',monospace;line-height:1;}
.metric-delta-up{font-size:12px;color:#f06c6c;margin-top:4px;}
.metric-delta-down{font-size:12px;color:#5eca8a;margin-top:4px;}
.metric-delta-flat{font-size:12px;color:#5a6070;margin-top:4px;}
.section-title{font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4055;margin:1.5rem 0 0.75rem;border-bottom:1px solid #1a1e28;padding-bottom:0.5rem;}
.advice-card{border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.5rem;}
.advice-buy-now{background:#2e1a1a;border-left:3px solid #f06c6c;}
.advice-buy-soon{background:#2e240e;border-left:3px solid #f0936c;}
.advice-wait{background:#1a2e1a;border-left:3px solid #5eca8a;}
.advice-watch{background:#1e2030;border-left:3px solid #6cb8f0;}
.advice-neutral{background:#1a1a1a;border-left:3px solid #5a6070;}
.headline-pill{background:#1a1e28;border-radius:6px;padding:4px 10px;font-size:11px;color:#5a6070;margin:3px 0;display:block;}
[data-testid="stSidebar"]{background:#0a0c10;border-right:1px solid #1a1e28;}
</style>""",unsafe_allow_html=True)

def fmt_value(v,unit):
    if pd.isna(v): return "—"
    if unit and "index" in unit.lower(): return f"{v:.1f}"
    if v>500: return f"${v:,.0f}"
    return f"${v:.2f}"

def delta_color(pct):
    if pct>2: return "metric-delta-up"
    if pct<-2: return "metric-delta-down"
    return "metric-delta-flat"

def delta_arrow(pct):
    if pct>0: return "▲"
    if pct<0: return "▼"
    return "—"

CATEGORY_COLORS={"Food & Beverages":"#5eca8a","Energy":"#f0936c","Housing":"#8c6cf0","Transportation":"#6cb8f0","Healthcare":"#f06c6c","Education":"#f0d96c","Apparel":"#f06cc8","Recreation":"#6cf0e8","Food Away From Home":"#a8f06c","Other":"#888"}
ADVICE_CLASS={"Buy now":"advice-buy-now","Buy soon":"advice-buy-soon","Wait":"advice-wait","Watch":"advice-watch","Neutral":"advice-neutral"}

@st.cache_data(ttl=3600)
def load_forecasts():
    p=Path("data/forecasts.csv")
    return pd.read_csv(p,parse_dates=["ds"]) if p.exists() else None

@st.cache_data(ttl=3600)
def load_metrics():
    p=Path("data/model_metrics.csv")
    return pd.read_csv(p,index_col="item") if p.exists() else None

@st.cache_data(ttl=3600)
def load_raw():
    p=Path("data/prices.csv")
    return pd.read_csv(p,parse_dates=["date"]) if p.exists() else None

@st.cache_data(ttl=86400)
def load_sentiment(api_key,items_tuple):
    if not api_key: return None
    try:
        from sentiment_engine import get_all_sentiment
        return get_all_sentiment(api_key,list(items_tuple))
    except Exception:
        return None

df_fc=load_forecasts()
df_raw=load_raw()

if df_fc is None:
    st.warning("Run python data_pipeline.py then python forecast_model.py first.")
    st.stop()

items=sorted(df_fc["item"].unique())
categories=sorted(df_fc["category"].unique())

with st.sidebar:
    st.markdown("## 📈 Inflation Tracker")
    st.markdown("<div class='section-title'>Filters</div>",unsafe_allow_html=True)
    selected_cat=st.selectbox("Category",["All"]+categories)
    filtered_items=items if selected_cat=="All" else [i for i in items if df_fc.loc[df_fc["item"]==i,"category"].iloc[0]==selected_cat]
    selected_item=st.selectbox("Item",filtered_items)
    years_back=st.slider("History (years)",1,10,5)
    st.markdown("<div class='section-title'>Live Intelligence</div>",unsafe_allow_html=True)
    api_key=st.text_input("NewsAPI key",type="password",value=os.environ.get("NEWS_API_KEY",""),help="Get free key at newsapi.org")
    st.markdown("<div class='section-title'>About</div>",unsafe_allow_html=True)
    st.caption("Data: BLS CPI-U · Forecast: ARIMA · Sentiment: NewsAPI + VADER")

tab1,tab2,tab3=st.tabs(["📊 Item deep-dive","🛒 Smart shopping advisor","🌍 Geopolitical risk index"])

sent_df=load_sentiment(api_key,tuple(items)) if api_key else None

with tab1:
    item_fc=df_fc[df_fc["item"]==selected_item].copy()
    item_unit=df_raw.loc[df_raw["item"]==selected_item,"unit"].iloc[0] if df_raw is not None else ""
    cutoff=item_fc["ds"].max()-pd.DateOffset(years=years_back)
    item_fc=item_fc[item_fc["ds"]>=cutoff]
    hist=item_fc[~item_fc["is_forecast"]]
    fcast=item_fc[item_fc["is_forecast"]]
    cur_val=hist["actual"].dropna().iloc[-1] if not hist.empty else np.nan
    prev_12=hist["actual"].dropna().iloc[-13] if len(hist)>=13 else hist["actual"].dropna().iloc[0]
    yoy_pct=(cur_val-prev_12)/prev_12*100 if prev_12 else 0
    prev_60=hist["actual"].dropna().iloc[-61] if len(hist)>=61 else hist["actual"].dropna().iloc[0]
    five_pct=(cur_val-prev_60)/prev_60*100 if prev_60 else 0
    fc_end=fcast["yhat"].iloc[-1] if not fcast.empty else np.nan
    fc_pct=(fc_end-cur_val)/cur_val*100 if cur_val else 0
    cat_color=CATEGORY_COLORS.get(item_fc["category"].iloc[0],"#888")
    item_sent=None
    if sent_df is not None and selected_item in sent_df["item"].values:
        item_sent=sent_df[sent_df["item"]==selected_item].iloc[0]

    st.markdown(f"""<div style='padding:1.5rem 0 0.5rem;'>
        <span style='font-size:28px;font-weight:600;color:#e8eaf0;'>{selected_item}</span>
        <span style='margin-left:12px;font-size:13px;color:{cat_color};background:{cat_color}22;padding:3px 10px;border-radius:20px;'>{item_fc["category"].iloc[0]}</span>
        {f'<span style="margin-left:8px;font-size:18px;">{item_sent["risk_emoji"]}</span>' if item_sent is not None else ''}
    </div>
    <div style='font-size:13px;color:#5a6070;margin-bottom:1.5rem;'>{item_unit} · BLS CPI-U · ARIMA forecast · NewsAPI + VADER sentiment</div>""",unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,label,val,delta,suffix in [(c1,"Current price",cur_val,None,""),(c2,"1-year change",yoy_pct,yoy_pct,"%"),(c3,"5-year change",five_pct,five_pct,"%"),(c4,"12-mo forecast",fc_end,fc_pct,"")]:
        with col:
            display=f"{val:+.1f}%" if suffix=="%" else fmt_value(val,item_unit)
            dh="" if delta is None or np.isnan(delta) else f"<div class='{delta_color(delta)}'>{delta_arrow(delta)} {abs(delta):.1f}%</div>"
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{display}</div>{dh}</div>",unsafe_allow_html=True)

    if item_sent is not None:
        from sentiment_engine import shopping_advice
        adv_label,adv_text=shopping_advice(item_sent["compound_score"],fc_pct)
        st.markdown(f"<br><div class='advice-card {ADVICE_CLASS.get(adv_label,'advice-neutral')}'><strong style='color:#e8eaf0;font-size:15px;'>{adv_label}</strong><span style='color:#8a90a0;font-size:13px;margin-left:10px;'>{adv_text}</span></div>",unsafe_allow_html=True)

    fig=go.Figure()
    if not fcast.empty:
        fig.add_trace(go.Scatter(x=pd.concat([fcast["ds"],fcast["ds"].iloc[::-1]]),y=pd.concat([fcast["yhat_upper"],fcast["yhat_lower"].iloc[::-1]]),fill="toself",fillcolor="rgba(239,147,108,0.12)",line=dict(width=0),name="95% confidence",hoverinfo="skip"))
        if not hist.empty:
            fig.add_trace(go.Scatter(x=[hist["ds"].iloc[-1],fcast["ds"].iloc[0]],y=[hist["actual"].iloc[-1],fcast["yhat"].iloc[0]],mode="lines",line=dict(color="#f0936c",width=2,dash="dot"),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fcast["ds"],y=fcast["yhat"],mode="lines",name="Forecast (ARIMA)",line=dict(color="#f0936c",width=2.5,dash="dot"),hovertemplate="%{x|%b %Y}<br>Forecast: %{y:.2f}<extra></extra>"))
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["ds"],y=hist["actual"],mode="lines",name="Actual price",line=dict(color=cat_color,width=2.5),hovertemplate="%{x|%b %Y}<br>Price: %{y:.2f}<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d0f14",font=dict(family="DM Sans",color="#5a6070"),margin=dict(l=10,r=10,t=10,b=10),height=360,legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,font=dict(size=12),bgcolor="rgba(0,0,0,0)"),xaxis=dict(showgrid=False,tickfont=dict(size=11),zeroline=False),yaxis=dict(gridcolor="#1a1e28",showgrid=True,tickfont=dict(size=11),tickprefix="" if "index" in item_unit.lower() else "$"),hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    col_a,col_b=st.columns([1,2])
    with col_a:
        st.markdown("<div class='section-title'>Model performance</div>",unsafe_allow_html=True)
        df_metrics=load_metrics()
        if df_metrics is not None and selected_item in df_metrics.index:
            for k,v in df_metrics.loc[selected_item].items():
                if v and not(isinstance(v,float) and np.isnan(v)):
                    st.metric(k,f"{v:.2f}%" if k=="MAPE_%" else f"{v:.4f}")
    with col_b:
        st.markdown("<div class='section-title'>Recent news headlines</div>",unsafe_allow_html=True)
        if item_sent is not None and item_sent["headline_count"]>0:
            st.caption(f"Sentiment: **{item_sent['compound_score']:+.3f}** · {item_sent['risk_emoji']} {item_sent['risk_level']} · {item_sent['headline_count']} headlines")
            for h in item_sent.get("headlines",[])[:5]:
                if h.strip(): st.markdown(f"<span class='headline-pill'>{h[:120]}...</span>",unsafe_allow_html=True)
        else:
            st.caption("Add your NewsAPI key in the sidebar to see live headlines and sentiment scoring.")

with tab2:
    st.markdown("<div style='padding:1rem 0 0.5rem;'><span style='font-size:24px;font-weight:600;color:#e8eaf0;'>Smart Shopping Advisor</span></div><div style='font-size:13px;color:#5a6070;margin-bottom:1.5rem;'>Combines ARIMA price forecasts + live news sentiment to tell you when to buy each item.</div>",unsafe_allow_html=True)
    advice_rows=[]
    for item in items:
        sub=df_fc[df_fc["item"]==item]
        hs=sub[~sub["is_forecast"]].dropna(subset=["actual"])
        fs=sub[sub["is_forecast"]]
        if hs.empty or fs.empty: continue
        cur=hs["actual"].iloc[-1]
        fc3=fs["yhat"].iloc[2] if len(fs)>=3 else fs["yhat"].iloc[-1]
        fc_p=(fc3-cur)/cur*100 if cur else 0
        cat=sub["category"].iloc[0]
        unit=sub["unit"].iloc[0] if "unit" in sub.columns else ""
        compound=0.0; risk_emoji="⚪"
        if sent_df is not None and item in sent_df["item"].values:
            rs=sent_df[sent_df["item"]==item].iloc[0]
            compound=rs["compound_score"]; risk_emoji=rs["risk_emoji"]
        from sentiment_engine import shopping_advice as sa
        adv_label,adv_text=sa(compound,fc_p)
        advice_rows.append({"item":item,"category":cat,"unit":unit,"current_price":fmt_value(cur,unit),"3mo_forecast_%":round(fc_p,1),"sentiment":risk_emoji,"advice":adv_label,"detail":adv_text})

    if advice_rows:
        df_advice=pd.DataFrame(advice_rows)
        f1,f2=st.columns([1,3])
        with f1: adv_filter=st.selectbox("Filter by",["All","Buy now","Buy soon","Wait","Watch","Neutral"])
        with f2: cat_filter=st.selectbox("Category",["All"]+categories,key="adv_cat")
        filtered=df_advice.copy()
        if adv_filter!="All": filtered=filtered[filtered["advice"]==adv_filter]
        if cat_filter!="All": filtered=filtered[filtered["category"]==cat_filter]
        s1,s2,s3,s4=st.columns(4)
        for col,label,color in [(s1,"Buy now","#f06c6c"),(s2,"Buy soon","#f0936c"),(s3,"Wait","#5eca8a"),(s4,"Watch","#6cb8f0")]:
            with col: st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{color};'>{len(df_advice[df_advice['advice']==label])} items</div></div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        for _,row in filtered.iterrows():
            fc_arrow="▲" if row["3mo_forecast_%"]>0 else "▼"
            fc_color="#f06c6c" if row["3mo_forecast_%"]>0 else "#5eca8a"
            advice_css = ADVICE_CLASS.get(row["advice"], "advice-neutral")
            st.markdown(f"<div class='advice-card {advice_css}'>{row['item']} — {row['detail']} — {row['current_price']}</div>", unsafe_allow_html=True)
    

with tab3:
    st.markdown("<div style='padding:1rem 0 0.5rem;'><span style='font-size:24px;font-weight:600;color:#e8eaf0;'>Geopolitical Risk Index</span></div><div style='font-size:13px;color:#5a6070;margin-bottom:1.5rem;'>How exposed is each category to current global events? Derived from live news sentiment across all tracked items.</div>",unsafe_allow_html=True)
    if sent_df is None:
        st.info("Add your NewsAPI key in the sidebar to activate the Geopolitical Risk Index.")
    else:
        merged=sent_df.merge(df_fc[["item","category"]].drop_duplicates(),on="item",how="left")
        cat_sent=merged.groupby("category").agg(avg_compound=("compound_score","mean"),item_count=("item","count"),high_risk=("risk_level",lambda x:(x=="HIGH").sum())).reset_index()
        cat_sent["risk_score"]=cat_sent["avg_compound"].apply(lambda x:round(-x*100,1))
        cat_sent=cat_sent.sort_values("risk_score",ascending=False)
        for _,row in cat_sent.iterrows():
            score=max(0,min(100,row["risk_score"]+50))
            color=CATEGORY_COLORS.get(row["category"],"#888")
            risk_l="HIGH" if row["avg_compound"]<-0.2 else "MEDIUM" if row["avg_compound"]<-0.05 else "LOW"
            emoji="🔴" if risk_l=="HIGH" else "🟡" if risk_l=="MEDIUM" else "🟢"
            st.markdown(f"<div style='margin-bottom:1rem;'><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='font-size:14px;color:#e8eaf0;'>{emoji} {row['category']}</span><span style='font-size:12px;color:#5a6070;'>{risk_l} · {row['item_count']} items</span></div><div style='background:#1a1e28;border-radius:4px;height:8px;'><div style='background:{color};width:{score}%;height:8px;border-radius:4px;'></div></div><div style='font-size:11px;color:#5a6070;margin-top:3px;'>Sentiment: {row['avg_compound']:+.3f} · {row['high_risk']} high-risk items</div></div>",unsafe_allow_html=True)

        scatter_rows=[]
        for item in items:
            sub=df_fc[df_fc["item"]==item]
            hs=sub[~sub["is_forecast"]].dropna(subset=["actual"])
            fs=sub[sub["is_forecast"]]
            if hs.empty or fs.empty: continue
            cur=hs["actual"].iloc[-1]; fc_e=fs["yhat"].iloc[-1]
            fc_p=(fc_e-cur)/cur*100 if cur else 0
            cat=sub["category"].iloc[0]
            compound=sent_df[sent_df["item"]==item]["compound_score"].iloc[0] if item in sent_df["item"].values else 0.0
            scatter_rows.append({"item":item,"category":cat,"fc_pct":round(fc_p,1),"compound":compound})

        if scatter_rows:
            sc_df=pd.DataFrame(scatter_rows)
            fig2=go.Figure()
            for cat in sc_df["category"].unique():
                sub2=sc_df[sc_df["category"]==cat]
                fig2.add_trace(go.Scatter(x=sub2["compound"],y=sub2["fc_pct"],mode="markers+text",name=cat,text=sub2["item"].str[:20],textposition="top center",textfont=dict(size=9,color="#5a6070"),marker=dict(size=10,color=CATEGORY_COLORS.get(cat,"#888"),opacity=0.85),hovertemplate="%{text}<br>Sentiment: %{x:.3f}<br>Forecast: %{y:.1f}%<extra></extra>"))
            fig2.add_vline(x=0,line_dash="dash",line_color="#2a2e3e",line_width=1)
            fig2.add_hline(y=0,line_dash="dash",line_color="#2a2e3e",line_width=1)
            fig2.add_annotation(x=-0.5,y=15,text="Negative sentiment + rising prices",font=dict(size=10,color="#f06c6c"),showarrow=False)
            fig2.add_annotation(x=0.3,y=-5,text="Positive sentiment + falling prices",font=dict(size=10,color="#5eca8a"),showarrow=False)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d0f14",font=dict(family="DM Sans",color="#5a6070"),margin=dict(l=10,r=10,t=20,b=10),height=420,xaxis=dict(title="News sentiment score",showgrid=True,gridcolor="#1a1e28",zeroline=False),yaxis=dict(title="12-month forecast change (%)",showgrid=True,gridcolor="#1a1e28",zeroline=False),legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,font=dict(size=11),bgcolor="rgba(0,0,0,0)"),hovermode="closest")
            st.plotly_chart(fig2,use_container_width=True)
            st.caption("Bottom-left = most at-risk: negative news AND rising prices.")
