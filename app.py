# -*- coding: utf-8 -*-
"""
OKKIO Loyalty Intelligence
Dashboard + RFM/K-Means + churn prediction (90 days)
Methodology aligned with OKKIO_Loyalty_Analytics_Colab.ipynb
"""
import io
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, roc_auc_score, average_precision_score,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lifelines import KaplanMeierFitter
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2026-06-30")
HORIZON_DAYS = 90

st.set_page_config(
    page_title="OKKIO | Loyalty Intelligence",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------- UI STYLE ----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: #171a1f; border: 1px solid #303640; padding: 15px 18px; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #aeb7c2;}
[data-testid="stMetricValue"] {color: #ffffff;}
h1, h2, h3 {letter-spacing: -0.02em;}
.small-muted {color:#9aa4b2; font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

# Basic access gate: required because the app displays customer phone numbers.
APP_PASSWORD = os.environ.get("OKKIO_APP_PASSWORD", "")
if not APP_PASSWORD:
    st.error("Website chưa được cấu hình mật khẩu truy cập. Hãy thiết lập biến môi trường OKKIO_APP_PASSWORD trước khi chạy.")
    st.stop()
if not st.session_state.get("authenticated", False):
    st.title("OKKIO Customer Care")
    st.markdown("### Đăng nhập")
    entered_password = st.text_input("Mật khẩu truy cập", type="password")
    if st.button("Đăng nhập", type="primary"):
        if entered_password == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mật khẩu chưa chính xác.")
    st.stop()

st.title("OKKIO Customer Care")
st.caption("Hội viên · Chăm sóc khách hàng · Marketing & Business")

# ---------------------------- HELPERS ----------------------------
def normalize_phone(value):
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value).split(".")[0] if ch.isdigit())
    if digits.startswith("84"):
        digits = "0" + digits[2:]
    return digits

def clean_source(xlsx_bytes):
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    required = {"Giao_dich", "Doi_diem", "TK_can_xem_xet"}
    missing = required - set(xls.sheet_names)
    if missing:
        raise ValueError(f"File thiếu sheet bắt buộc: {', '.join(sorted(missing))}")

    tx = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="Giao_dich")
    red = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="Doi_diem")
    flagged = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="TK_can_xem_xet")
    contacts = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="Danh_ba_hoi_vien")
    required_contacts = ["Mã hội viên", "SĐT", "Tên hội viên"]
    missing_contacts = [c for c in required_contacts if c not in contacts.columns]
    if missing_contacts:
        raise ValueError(f"Sheet Danh_ba_hoi_vien thiếu cột: {', '.join(missing_contacts)}")
    contacts["Mã hội viên"] = contacts["Mã hội viên"].astype(str).str.strip()
    contacts["SĐT"] = contacts["SĐT"].apply(normalize_phone)
    contacts["Tên hội viên"] = contacts["Tên hội viên"].fillna("").astype(str).str.strip()

    required_tx = ["Mã hội viên", "Thời gian", "ID giao dịch",
                   "Tiền chi tiêu (VND)", "Chi nhánh", "Ngày",
                   "Có dùng mã giảm giá", "HĐ 0đ"]
    missing_tx = [c for c in required_tx if c not in tx.columns]
    if missing_tx:
        raise ValueError(f"Sheet Giao_dich thiếu cột: {', '.join(missing_tx)}")

    tx["Thời gian"] = pd.to_datetime(tx["Thời gian"], errors="coerce")
    red["Thời gian"] = pd.to_datetime(red["Thời gian"], errors="coerce")
    tx["Tiền chi tiêu (VND)"] = pd.to_numeric(tx["Tiền chi tiêu (VND)"], errors="coerce").fillna(0)
    tx["Điểm cộng"] = pd.to_numeric(tx.get("Điểm cộng", 0), errors="coerce").fillna(0)
    if "Điểm" in red:
        red["Điểm"] = pd.to_numeric(red["Điểm"], errors="coerce").fillna(0)
    else:
        red["Điểm"] = 0

    tx = tx.dropna(subset=["Mã hội viên", "Thời gian"]).copy()
    red = red.dropna(subset=["Mã hội viên", "Thời gian"]).copy()
    tx["Mã hội viên"] = tx["Mã hội viên"].astype(str)
    red["Mã hội viên"] = red["Mã hội viên"].astype(str)
    tx["is_zero_bill"] = (tx["Tiền chi tiêu (VND)"] == 0).astype(int)
    tx["is_paid_bill"] = (tx["Tiền chi tiêu (VND)"] > 0).astype(int)
    tx["is_discount_used"] = pd.to_numeric(tx["Có dùng mã giảm giá"], errors="coerce").fillna(0).astype(int)

    return tx, red, flagged, contacts, xls.sheet_names


def aggregate_customer(tx, red, as_of):
    hist = tx[tx["Thời gian"] <= as_of].copy()
    hist_red = red[red["Thời gian"] <= as_of].copy()
    if hist.empty:
        raise ValueError("Không có giao dịch trước hoặc tại ngày phân tích.")

    cust = hist.groupby("Mã hội viên").agg(
        first_purchase=("Thời gian", "min"),
        last_purchase=("Thời gian", "max"),
        frequency=("ID giao dịch", "nunique"),
        monetary=("Tiền chi tiêu (VND)", "sum"),
        paid_frequency=("is_paid_bill", "sum"),
        zero_bill_count=("is_zero_bill", "sum"),
        discount_bill_count=("Có dùng mã giảm giá", "sum"),
        avg_bill=("Tiền chi tiêu (VND)", "mean"),
        median_bill=("Tiền chi tiêu (VND)", "median"),
        branch_count=("Chi nhánh", "nunique"),
        active_days=("Ngày", "nunique")
    ).reset_index()

    if not hist_red.empty and "Loại nhật ký" in hist_red.columns:
        ra = hist_red.groupby("Mã hội viên").agg(
            redemption_log_count=("Điểm", "size"),
            points_net=("Điểm", "sum"),
            points_redeemed=("Điểm", lambda x: -x[x < 0].sum()),
            manual_adjustment_count=("Loại nhật ký", lambda x: (x == "Thay đổi điểm bằng tay").sum()),
            redemption_count=("Loại nhật ký", lambda x: (x == "Người dùng đổi điểm").sum())
        ).reset_index()
        cust = cust.merge(ra, on="Mã hội viên", how="left")
    else:
        for c in ["redemption_log_count", "points_net", "points_redeemed",
                  "manual_adjustment_count", "redemption_count"]:
            cust[c] = 0

    red_cols = ["redemption_log_count", "points_net", "points_redeemed",
                "manual_adjustment_count", "redemption_count"]
    cust[red_cols] = cust[red_cols].fillna(0)
    cust["recency_days"] = (as_of.normalize() - cust["last_purchase"].dt.normalize()).dt.days
    cust["tenure_days"] = (as_of.normalize() - cust["first_purchase"].dt.normalize()).dt.days
    cust["frequency"] = cust["frequency"].astype(int)
    cust["monetary"] = cust["monetary"].astype(float)
    return cust


def make_model_frame(tx, red, snapshot, horizon=90):
    hist_tx = tx[tx["Thời gian"] <= snapshot].copy()
    future_tx = tx[(tx["Thời gian"] > snapshot) &
                   (tx["Thời gian"] <= snapshot + pd.Timedelta(days=horizon))].copy()
    hist_red = red[red["Thời gian"] <= snapshot].copy()

    base = hist_tx.groupby("Mã hội viên").agg(
        first_purchase=("Thời gian", "min"),
        last_purchase=("Thời gian", "max"),
        frequency=("ID giao dịch", "nunique"),
        monetary=("Tiền chi tiêu (VND)", "sum"),
        paid_frequency=("is_paid_bill", "sum"),
        zero_bill_count=("HĐ 0đ", "sum"),
        discount_bill_count=("Có dùng mã giảm giá", "sum"),
        avg_bill=("Tiền chi tiêu (VND)", "mean"),
        branch_count=("Chi nhánh", "nunique"),
        active_days=("Ngày", "nunique")
    ).reset_index()

    base["recency_days"] = (snapshot - base["last_purchase"].dt.normalize()).dt.days
    base["tenure_days"] = (snapshot - base["first_purchase"].dt.normalize()).dt.days

    for days in [30,  60, 90, 180]:
        w = hist_tx[hist_tx["Thời gian"] > snapshot - pd.Timedelta(days=days)]
        agg = w.groupby("Mã hội viên").agg(
            **{f"bills_{days}d": ("ID giao dịch", "nunique"),
               f"spend_{days}d": ("Tiền chi tiêu (VND)", "sum"),
               f"zero_{days}d": ("HĐ 0đ", "sum")}
        ).reset_index()
        base = base.merge(agg, on="Mã hội viên", how="left")

    base = base.fillna(0)

    if not hist_red.empty and "Loại nhật ký" in hist_red.columns:
        ra = hist_red.groupby("Mã hội viên").agg(
            redemption_count=("Loại nhật ký", lambda x: (x == "Người dùng đổi điểm").sum()),
            points_redeemed=("Điểm", lambda x: -x[x < 0].sum()),
            manual_adjustments=("Loại nhật ký", lambda x: (x == "Thay đổi điểm bằng tay").sum())
        ).reset_index()
        base = base.merge(ra, on="Mã hội viên", how="left")
    else:
        base["redemption_count"] = 0
        base["points_redeemed"] = 0
        base["manual_adjustments"] = 0

    for c in ["redemption_count", "points_redeemed", "manual_adjustments"]:
        base[c] = base[c].fillna(0)

    future_active = set(future_tx["Mã hội viên"].astype(str).unique())
    base["churn_90d"] = (~base["Mã hội viên"].astype(str).isin(future_active)).astype(int)
    return base


def current_scoring_frame(tx, red, as_of):
    # Re-create the same feature columns used by the research model, without a future label.
    hist_tx = tx[tx["Thời gian"] <= as_of].copy()
    hist_red = red[red["Thời gian"] <= as_of].copy()
    base = hist_tx.groupby("Mã hội viên").agg(
        first_purchase=("Thời gian", "min"),
        last_purchase=("Thời gian", "max"),
        frequency=("ID giao dịch", "nunique"),
        monetary=("Tiền chi tiêu (VND)", "sum"),
        paid_frequency=("is_paid_bill", "sum"),
        zero_bill_count=("HĐ 0đ", "sum"),
        discount_bill_count=("Có dùng mã giảm giá", "sum"),
        avg_bill=("Tiền chi tiêu (VND)", "mean"),
        branch_count=("Chi nhánh", "nunique"),
        active_days=("Ngày", "nunique")
    ).reset_index()
    base["recency_days"] = (as_of - base["last_purchase"].dt.normalize()).dt.days
    base["tenure_days"] = (as_of - base["first_purchase"].dt.normalize()).dt.days
    for days in [30, 60, 90, 180]:
        w = hist_tx[hist_tx["Thời gian"] > as_of - pd.Timedelta(days=days)]
        agg = w.groupby("Mã hội viên").agg(
            **{f"bills_{days}d": ("ID giao dịch", "nunique"),
               f"spend_{days}d": ("Tiền chi tiêu (VND)", "sum"),
               f"zero_{days}d": ("HĐ 0đ", "sum")}
        ).reset_index()
        base = base.merge(agg, on="Mã hội viên", how="left")
    base = base.fillna(0)
    if not hist_red.empty and "Loại nhật ký" in hist_red.columns:
        ra = hist_red.groupby("Mã hội viên").agg(
            redemption_count=("Loại nhật ký", lambda x: (x == "Người dùng đổi điểm").sum()),
            points_redeemed=("Điểm", lambda x: -x[x < 0].sum()),
            manual_adjustments=("Loại nhật ký", lambda x: (x == "Thay đổi điểm bằng tay").sum())
        ).reset_index()
        base = base.merge(ra, on="Mã hội viên", how="left")
    else:
        base["redemption_count"] = 0
        base["points_redeemed"] = 0
        base["manual_adjustments"] = 0
    for c in ["redemption_count", "points_redeemed", "manual_adjustments"]:
        base[c] = base[c].fillna(0)
    return base


def train_models(model_data):
    exclude = ["Mã hội viên", "first_purchase", "last_purchase", "churn_90d"]
    features = [c for c in model_data.columns if c not in exclude]
    X = model_data[features].select_dtypes(include=[np.number]).copy()
    y = model_data["churn_90d"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=450, max_depth=3, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
            min_child_weight=5, objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=2
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=450, max_depth=-1, num_leaves=15,
            learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2, class_weight="balanced",
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=2
        )
    }
    fitted, rows, reports = {}, [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        rows.append({
            "Mô hình": name,
            "ROC-AUC": roc_auc_score(y_test, prob),
            "PR-AUC": average_precision_score(y_test, prob),
            "Accuracy": accuracy_score(y_test, pred)
        })
        reports[name] = {
            "cm": confusion_matrix(y_test, pred),
            "report": classification_report(y_test, pred, output_dict=True),
            "y_test": y_test, "prob": prob
        }
    scores = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    best_name = scores.iloc[0]["Mô hình"]
    # Refit selected algorithm on the full labelled cohort for current-customer scoring.
    best_model = models[best_name]
    best_model.fit(X, y)
    return features, scores, reports, best_name, best_model, X, y


def fmt_money(v):
    return f"{v:,.0f} ₫"


def risk_band(p):
    if p >= 0.70:
        return "Cao"
    if p >= 0.40:
        return "Trung bình"
    return "Thấp"


def make_cluster(cust):
    rfm = cust[["Mã hội viên", "recency_days", "frequency", "monetary"]].copy()
    X = rfm[["recency_days", "frequency", "monetary"]].copy()
    X["recency_days"] = np.log1p(X["recency_days"].clip(lower=0))
    X["frequency"] = np.log1p(X["frequency"].clip(lower=0))
    X["monetary"] = np.log1p(X["monetary"].clip(lower=0))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(Xs)
        scores.append({"k": k, "silhouette": silhouette_score(Xs, labels), "inertia": km.inertia_})
    score_df = pd.DataFrame(scores)
    best_k = int(score_df.loc[score_df["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30)
    cust = cust.copy()
    cust["cluster"] = km.fit_predict(Xs)
    profile = cust.groupby("cluster").agg(
        members=("Mã hội viên", "nunique"),
        recency_median=("recency_days", "median"),
        frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"),
        monetary_mean=("monetary", "mean"),
        zero_bill_mean=("zero_bill_count", "mean"),
        redemption_rate=("redemption_count", lambda x: (x > 0).mean())
    ).reset_index()
    return cust, score_df, profile, best_k


# ---------------------------- DATA LOAD ----------------------------
with st.sidebar:
    st.header("Dữ liệu hội viên")
    upload = st.file_uploader("Tải file OKKIO loyalty (.xlsx)", type=["xlsx"])
    st.caption("Tra cứu bằng số điện thoại đăng ký. Thông tin liên hệ chỉ dành cho nhân sự được phân quyền.")
    st.divider()
    st.caption("Dữ liệu hội viên và lịch sử giao dịch được xử lý trong phiên chạy hiện tại.")

if upload is None:
    st.info("Tải file Excel loyalty để khởi chạy dashboard và mô hình dự báo.")
    st.markdown("""
    **Website cung cấp**
    - Tra cứu hội viên bằng số điện thoại đăng ký
    - Danh sách cần chăm sóc, cửa hàng phụ trách và kênh liên hệ đề xuất
    - Lịch sử giao dịch và cửa hàng gần nhất
    - Phân khúc khách hàng và gợi ý hoạt động Marketing / Business
    """)
    st.stop()

file_bytes = upload.getvalue()
file_signature = f"{upload.name}-{len(file_bytes)}"
if st.session_state.get("file_signature") != file_signature:
    with st.spinner("Đang đọc dữ liệu, tái tạo phân tích và huấn luyện mô hình..."):
        try:
            tx, red, flagged, contacts, sheet_names = clean_source(file_bytes)
            data_end = tx["Thời gian"].max().normalize()
            if data_end < SNAPSHOT_DATE + pd.Timedelta(days=HORIZON_DAYS):
                raise ValueError(
                    f"Dữ liệu kết thúc {data_end.date()}, chưa đủ 90 ngày theo dõi sau snapshot "
                    f"{SNAPSHOT_DATE.date()}. Cần dữ liệu ít nhất đến {(SNAPSHOT_DATE + pd.Timedelta(days=90)).date()}."
                )
            model_data = make_model_frame(tx, red, SNAPSHOT_DATE, HORIZON_DAYS)
            cust = aggregate_customer(tx, red, data_end)
            cust, km_scores, cluster_profile, best_k = make_cluster(cust)
            features, model_scores, reports, best_name, best_model, X_all, y_all = train_models(model_data)
            scoring = current_scoring_frame(tx, red, data_end)
            # Align current features exactly to training feature schema.
            X_current = scoring.reindex(columns=features, fill_value=0).select_dtypes(include=[np.number])
            X_current = X_current.reindex(columns=features, fill_value=0)
            scoring["churn_probability"] = best_model.predict_proba(X_current)[:, 1]
            scoring["risk_level"] = scoring["churn_probability"].apply(risk_band)
            scoring["risk_score"] = (scoring["churn_probability"] * 100).round(1)
            scoring = scoring.merge(cust[["Mã hội viên", "cluster"]], on="Mã hội viên", how="left")
            scoring = scoring.merge(cust[["Mã hội viên", "recency_days", "frequency", "monetary",
                                          "paid_frequency", "zero_bill_count", "branch_count",
                                          "active_days", "redemption_count", "points_redeemed"]],
                                    on="Mã hội viên", how="left", suffixes=("", "_profile"))

            # Last known store: lấy từ giao dịch mới nhất, không suy đoán từ chi nhánh thường ghé.
            latest_tx = (tx.sort_values("Thời gian")
                           .drop_duplicates("Mã hội viên", keep="last")
                           [["Mã hội viên", "Thời gian", "Chi nhánh", "Mã cửa hàng"]]
                           .rename(columns={"Thời gian":"last_visit_at",
                                            "Chi nhánh":"last_store",
                                            "Mã cửa hàng":"last_store_code"}))
            scoring = scoring.merge(latest_tx, on="Mã hội viên", how="left")
            scoring = scoring.merge(contacts[["Mã hội viên", "SĐT", "Tên hội viên"]],
                                    on="Mã hội viên", how="left")
            scoring["SĐT"] = scoring["SĐT"].fillna("")
            scoring["Tên hội viên"] = scoring["Tên hội viên"].fillna("")
            scoring["Xử lý"] = np.select(
                [scoring["risk_level"].eq("Cao"), scoring["risk_level"].eq("Trung bình")],
                ["ƯU TIÊN: cửa hàng gần nhất chủ động hỏi thăm trong 24–48h",
                 "Theo dõi: cửa hàng lưu ý ở lần ghé tiếp theo"],
                default="Duy trì: chăm sóc theo lịch hội viên"
            )
            scoring["Marketing"] = np.where(
                scoring["SĐT"].astype(str).str.len().ge(9),
                np.select(
                    [scoring["risk_level"].eq("Cao"), scoring["risk_level"].eq("Trung bình")],
                    ["Liên hệ cá nhân hóa qua Zalo/SMS/điện thoại sau khi xác nhận đồng ý nhận tin",
                     "Gửi nội dung quay lại phù hợp qua Zalo/SMS sau khi xác nhận đồng ý nhận tin"],
                    default="Gửi nội dung duy trì tương tác sau khi xác nhận đồng ý nhận tin"
                ),
                "Chưa có SĐT hợp lệ – không gửi tin"
            )
            st.session_state["payload"] = {
                "tx": tx, "red": red, "flagged": flagged, "contacts": contacts, "data_end": data_end,
                "cust": cust, "km_scores": km_scores, "cluster_profile": cluster_profile,
                "best_k": best_k, "model_data": model_data, "features": features,
                "model_scores": model_scores, "reports": reports, "best_name": best_name,
                "best_model": best_model, "scoring": scoring
            }
            st.session_state["file_signature"] = file_signature
        except Exception as e:
            st.error(f"Không thể xử lý dữ liệu: {e}")
            st.stop()

p = st.session_state["payload"]
tx, red = p["tx"], p["red"]
scoring = p["scoring"].copy()
cust = p["cust"]
data_end = p["data_end"]

# ---------------------------- NAVIGATION ----------------------------
with st.sidebar:
    page = st.radio("Điều hướng", [
        "Tổng quan", "Tra cứu hội viên", "Danh sách cần chăm sóc",
        "Nhóm khách hàng", "Marketing & Business"
    ])
    st.divider()
    st.caption(f"Dữ liệu cập nhật đến: {data_end.date()}")

# ---------------------------- PAGE: OVERVIEW ----------------------------
if page == "Tổng quan":
    st.subheader("Tổng quan chương trình hội viên")
    total_bills = tx["ID giao dịch"].nunique()
    total_members = tx["Mã hội viên"].nunique()
    revenue = tx["Tiền chi tiêu (VND)"].sum()
    zero_bills = int((tx["Tiền chi tiêu (VND)"] == 0).sum())
    risk_high = int((scoring["risk_level"] == "Cao").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hội viên", f"{total_members:,}")
    c2.metric("Hóa đơn", f"{total_bills:,}")
    c3.metric("Chi tiêu ghi nhận", fmt_money(revenue))
    c4.metric("Hóa đơn 0 đồng", f"{zero_bills:,}")
    c5.metric("Rủi ro cao", f"{risk_high:,}")

    left, right = st.columns([1.35, 1])
    with left:
        monthly = tx.assign(Tháng=tx["Thời gian"].dt.to_period("M").astype(str)).groupby("Tháng").agg(
            doanh_thu=("Tiền chi tiêu (VND)", "sum"),
            so_hoa_don=("ID giao dịch", "nunique")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["Tháng"], y=monthly["doanh_thu"], name="Chi tiêu"))
        fig.update_layout(title="Chi tiêu theo tháng", xaxis_title="", yaxis_title="VND",
                          height=360, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        risk_counts = scoring["risk_level"].value_counts().reindex(["Cao","Trung bình","Thấp"]).fillna(0).reset_index()
        risk_counts.columns = ["Mức rủi ro", "Số hội viên"]
        fig = px.pie(risk_counts, names="Mức rủi ro", values="Số hội viên",
                     hole=.58, title="Phân bổ mức rủi ro dự báo")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Hội viên cần ưu tiên chăm sóc")
    priority = scoring.sort_values("churn_probability", ascending=False).head(12)
    st.dataframe(priority[["Tên hội viên","SĐT","risk_score","risk_level","last_store",
                           "last_visit_at","Xử lý","Marketing"]].rename(columns={
        "risk_score":"Điểm rủi ro (%)", "risk_level":"Mức rủi ro",
        "last_store":"Cửa hàng gần nhất", "last_visit_at":"Lần ghé gần nhất"
    }), use_container_width=True, hide_index=True)

# ---------------------------- PAGE: MEMBERS ----------------------------
elif page == "Tra cứu hội viên":
    st.subheader("Tra cứu và phân tích hội viên")
    search = st.text_input("Nhập số điện thoại đăng ký", placeholder="Ví dụ: 09xxxxxxxx")
    col1, col2, col3 = st.columns(3)
    risk_filter = col1.selectbox("Mức rủi ro", ["Tất cả","Cao","Trung bình","Thấp"])
    cluster_options = ["Tất cả"] + sorted(scoring["cluster"].dropna().astype(int).unique().tolist())
    cluster_filter = col2.selectbox("Cụm RFM", cluster_options)
    min_prob = col3.slider("Điểm rủi ro tối thiểu (%)", 0, 100, 0)
    view = scoring.copy()
    if search:
        normalized_query = normalize_phone(search)
        view = view[
            view["SĐT"].astype(str).str.contains(normalized_query, na=False) |
            view["Mã hội viên"].astype(str).str.contains(search.strip(), case=False, na=False)
        ]
    if risk_filter != "Tất cả":
        view = view[view["risk_level"] == risk_filter]
    if cluster_filter != "Tất cả":
        view = view[view["cluster"] == cluster_filter]
    view = view[view["risk_score"] >= min_prob].sort_values("churn_probability", ascending=False)
    st.caption(f"{len(view):,} hội viên phù hợp bộ lọc")
    display_cols = ["Tên hội viên","SĐT","risk_score","risk_level","last_store",
                    "last_store_code","last_visit_at","Xử lý","Marketing","recency_days",
                    "frequency","monetary","cluster"]
    st.dataframe(view[display_cols].rename(columns={
        "risk_score":"Điểm rủi ro (%)", "risk_level":"Mức rủi ro", "cluster":"Nhóm khách",
        "last_store":"Cửa hàng gần nhất", "last_store_code":"Mã cửa hàng",
        "last_visit_at":"Lần ghé gần nhất",
        "recency_days":"Số ngày chưa quay lại", "frequency":"Số hóa đơn",
        "monetary":"Tổng chi tiêu (VND)"
    }), use_container_width=True, hide_index=True)

    st.markdown("#### Hồ sơ hội viên")
    ids = view["Mã hội viên"].astype(str).tolist()
    if ids:
        member_id = st.selectbox(
            "Chọn hội viên",
            ids,
            format_func=lambda mid: (
                str(scoring.loc[scoring["Mã hội viên"].astype(str).eq(mid), "SĐT"].iloc[0])
                + " · " + str(scoring.loc[scoring["Mã hội viên"].astype(str).eq(mid), "Tên hội viên"].iloc[0])
            )
        )
        row = scoring[scoring["Mã hội viên"].astype(str) == member_id].iloc[0]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Điểm cần lưu ý", f"{row['risk_score']:.1f}%")
        m2.metric("Mức ưu tiên", row["risk_level"])
        m3.metric("Tổng chi tiêu", fmt_money(row["monetary"]))
        m4.metric("Số hóa đơn", f"{int(row['frequency']):,}")
        st.write(f"**Hội viên:** {row['Tên hội viên'] or 'Chưa có tên'} · **SĐT:** {row['SĐT'] or 'Chưa có SĐT'}")
        st.write(f"**Cửa hàng phụ trách hỏi thăm:** {row['last_store']} · **Lần ghé:** {row['last_visit_at']}")
        st.write(f"**Xử lý:** {row['Xử lý']}")
        st.write(f"**Marketing:** {row['Marketing']}")
        member_tx = tx[tx["Mã hội viên"].astype(str) == member_id].sort_values("Thời gian", ascending=False)
        st.write(f"**Lần mua gần nhất:** {row['last_purchase'].date() if 'last_purchase' in row and pd.notna(row['last_purchase']) else member_tx['Thời gian'].max().date()}")
        st.write(f"**Số ngày từ lần mua gần nhất:** {int(row['recency_days'])} ngày · **Cụm RFM:** {int(row['cluster'])}")
        st.dataframe(member_tx[["Thời gian","ID giao dịch","Chi nhánh","Tiền chi tiêu (VND)",
                                "Có dùng mã giảm giá","HĐ 0đ"]].head(100),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Không có hội viên phù hợp bộ lọc.")

    csv = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Tải danh sách hội viên đang lọc (CSV)", csv,
                       "okkio_member_scores.csv", "text/csv")

# ---------------------------- PAGE: CARE WORKLIST ----------------------------
elif page == "Danh sách cần chăm sóc":
    st.subheader("Danh sách chăm sóc hội viên")
    st.caption("Danh sách hành động để phân công cho cửa hàng và Marketing. Đây là gợi ý công việc, chưa tự động gửi tin hoặc tạo ticket.")
    care = scoring[scoring["risk_level"].isin(["Cao", "Trung bình"])].copy()
    c1, c2, c3 = st.columns(3)
    care_level = c1.selectbox("Mức ưu tiên", ["Tất cả", "Cao", "Trung bình"])
    store_opts = ["Tất cả"] + sorted(care["last_store"].dropna().astype(str).unique().tolist())
    store_filter = c2.selectbox("Cửa hàng phụ trách", store_opts)
    channel_filter = c3.selectbox("Kênh Marketing", ["Tất cả", "Có SĐT", "Thiếu SĐT"])
    if care_level != "Tất cả":
        care = care[care["risk_level"] == care_level]
    if store_filter != "Tất cả":
        care = care[care["last_store"] == store_filter]
    if channel_filter == "Có SĐT":
        care = care[care["SĐT"].astype(str).str.len() >= 9]
    elif channel_filter == "Thiếu SĐT":
        care = care[care["SĐT"].astype(str).str.len() < 9]
    care = care.sort_values("churn_probability", ascending=False)
    st.metric("Số hội viên trong danh sách", f"{len(care):,}")
    care_cols = ["Tên hội viên","SĐT","Mã hội viên","risk_score","risk_level",
                 "last_store","last_store_code","last_visit_at","Xử lý","Marketing"]
    st.dataframe(care[care_cols].rename(columns={
        "risk_score":"Điểm rủi ro (%)","risk_level":"Ưu tiên",
        "last_store":"Cửa hàng nhận thông tin","last_store_code":"Mã cửa hàng",
        "last_visit_at":"Lần ghé gần nhất"
    }), use_container_width=True, hide_index=True)
    st.download_button(
        "Xuất danh sách phân công cửa hàng & Marketing (Excel)",
        care[care_cols].to_csv(index=False).encode("utf-8-sig"),
        "okkio_customer_care_worklist.csv", "text/csv"
    )
    st.info("Quy trình đề xuất: danh sách được chia theo cửa hàng ghi nhận ở giao dịch gần nhất để quản lý cửa hàng phân công nhân viên lưu ý/hỏi thăm khi phù hợp. Marketing phụ trách liên hệ qua Zalo/SMS/điện thoại nếu có SĐT hợp lệ và khách đã đồng ý nhận tin. Website hiện xuất danh sách để bàn giao, chưa tự động gửi tin hoặc đẩy dữ liệu vào iPOS/Zalo.")

# ---------------------------- PAGE: RFM ----------------------------
elif page == "Nhóm khách hàng":
    st.subheader("Nhóm hành vi hội viên")
    st.caption("Các nhóm được hình thành từ hành vi gần đây, tần suất ghé và tổng chi tiêu.")
    st.dataframe(p["cluster_profile"].rename(columns={
        "cluster":"Cụm", "members":"Số hội viên", "recency_median":"Recency trung vị (ngày)",
        "frequency_median":"Frequency trung vị", "monetary_median":"Monetary trung vị",
        "monetary_mean":"Monetary trung bình", "zero_bill_mean":"Hóa đơn 0đ trung bình",
        "redemption_rate":"Tỷ lệ có đổi điểm"
    }), use_container_width=True, hide_index=True)
    fig = px.scatter(cust, x="recency_days", y="monetary", color=cust["cluster"].astype(str),
                     hover_data=["Mã hội viên","frequency"], log_y=True,
                     labels={"recency_days":"Recency (ngày)","monetary":"Tổng chi tiêu (VND)","color":"Cụm"})
    fig.update_layout(height=520, title="Phân bố hội viên theo Recency và Monetary")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Tên nhóm là nhóm hành vi để hỗ trợ phân công chăm sóc; không nên xem một nhóm là tốt/xấu tuyệt đối.")

# ---------------------------- PAGE: MARKETING & BUSINESS ----------------------------
elif page == "Marketing & Business":
    st.subheader("Marketing & Business Insights")
    st.caption("Các gợi ý bên dưới được rút ra từ hành vi giao dịch quan sát được; cần thử nghiệm trước khi kết luận hiệu quả gia tăng.")

    st.markdown("#### 1. Hiệu quả theo cửa hàng")
    branch = tx.groupby("Chi nhánh").agg(
        so_hoa_don=("ID giao dịch","nunique"),
        so_hoi_vien=("Mã hội viên","nunique"),
        chi_tieu=("Tiền chi tiêu (VND)","sum"),
        hoa_don_0d=("HĐ 0đ","sum")
    ).reset_index().sort_values("chi_tieu",ascending=False)
    branch["chi_tieu_tb_hoa_don"] = branch["chi_tieu"] / branch["so_hoa_don"].clip(lower=1)
    st.dataframe(branch.rename(columns={
        "Chi nhánh":"Cửa hàng","so_hoa_don":"Số hóa đơn","so_hoi_vien":"Số hội viên",
        "chi_tieu":"Chi tiêu ghi nhận (VND)","hoa_don_0d":"Hóa đơn 0đ",
        "chi_tieu_tb_hoa_don":"Chi tiêu TB / hóa đơn"
    }),use_container_width=True,hide_index=True)

    st.markdown("#### 2. Xu hướng khách hàng và chi tiêu")
    monthly = tx.assign(Tháng=tx["Thời gian"].dt.to_period("M").astype(str)).groupby("Tháng").agg(
        chi_tieu=("Tiền chi tiêu (VND)","sum"),
        hoa_don=("ID giao dịch","nunique"),
        hoi_vien=("Mã hội viên","nunique")
    ).reset_index()
    fig = px.line(monthly,x="Tháng",y=["chi_tieu","hoa_don","hoi_vien"],
                  title="Xu hướng theo tháng",markers=True)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### 3. Cơ hội Marketing theo nhóm hành vi")
    segment = scoring.groupby("cluster").agg(
        members=("Mã hội viên","nunique"),
        avg_risk=("churn_probability","mean"),
        median_recency=("recency_days","median"),
        median_frequency=("frequency","median"),
        median_spend=("monetary","median"),
        phone_coverage=("SĐT",lambda x:(x.astype(str).str.len()>=9).mean())
    ).reset_index()
    segment["Định hướng"] = segment.apply(
        lambda r: ("Win-back: cá nhân hóa lời mời quay lại, ưu tiên liên hệ trực tiếp"
                  if r["avg_risk"]>=0.7 else
                  "Tăng nhịp ghé: nhắc quyền lợi, gợi ý dịp ghé tiếp theo"
                  if r["median_recency"]>30 else
                  "Loyalty: ghi nhận, trải nghiệm ưu tiên và giới thiệu sản phẩm mới"),
        axis=1
    )
    st.dataframe(segment.rename(columns={
        "cluster":"Nhóm hành vi","members":"Số hội viên","avg_risk":"Rủi ro TB",
        "median_recency":"Recency trung vị (ngày)","median_frequency":"Số hóa đơn trung vị",
        "median_spend":"Chi tiêu trung vị (VND)","phone_coverage":"Tỷ lệ có SĐT"
    }),use_container_width=True,hide_index=True)

    st.markdown("#### 4. Cơ hội từ dữ liệu hành vi")
    one_day_members = int((cust["active_days"] == 1).sum())
    repeat_members = int((cust["active_days"] >= 2).sum())
    promo_member_count = int(tx.loc[tx["Có dùng mã giảm giá"].eq(1), "Mã hội viên"].nunique())
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Chỉ ghé 1 ngày", f"{one_day_members:,}", f"{one_day_members/max(1,len(cust)):.1%} hội viên")
    v2.metric("Đã ghé từ 2 ngày", f"{repeat_members:,}")
    v3.metric("Hội viên từng dùng mã", f"{promo_member_count:,}")
    v4.metric("Hóa đơn có mã giảm giá", f"{tx['Có dùng mã giảm giá'].mean():.1%}")

    first_visit = cust[cust["active_days"] == 1].merge(
        scoring[["Mã hội viên","SĐT","Tên hội viên","risk_score","risk_level","last_store","Marketing"]],
        on="Mã hội viên", how="left"
    ).sort_values("risk_score", ascending=False).head(20)
    st.markdown("**Nhóm mới/ít quay lại – gợi ý ưu tiên chuyển đổi lần ghé thứ hai**")
    st.dataframe(first_visit[["Tên hội viên","SĐT","risk_score","last_store","Marketing"]].rename(columns={
        "risk_score":"Điểm cần lưu ý (%)","last_store":"Cửa hàng gần nhất"
    }), use_container_width=True, hide_index=True)

    st.markdown("#### 5. Các hướng ứng dụng kinh doanh")
    st.markdown("""
    - **Win-back:** ưu tiên nhóm điểm rủi ro cao; chia danh sách theo cửa hàng ghé gần nhất để nhân viên hỏi thăm và Marketing gửi nội dung cá nhân hóa.
    - **First-to-second visit:** nhóm mới/ít lần ghé cần chuỗi nhắc quay lại trong vài tuần đầu; đo tỷ lệ phát sinh hóa đơn thứ hai.
    - **Loyalty & frequency:** nhóm ghé đều nên thử quyền lợi theo tần suất/hạng, trải nghiệm ưu tiên hoặc phần thưởng phi giảm giá.
    - **Branch recovery:** nếu cửa hàng khách ghé gần nhất ngừng hoạt động/đổi địa điểm, chuyển thông điệp sang cửa hàng đang hoạt động phù hợp; cần duy trì bảng mapping trạng thái cửa hàng.
    - **Promotion effectiveness:** so sánh nhóm có/không dùng mã giảm giá về tần suất và chi tiêu; đây là tương quan quan sát, không chứng minh voucher gây ra việc quay lại.
    - **Store operations:** dùng số khách có rủi ro cao theo cửa hàng làm danh sách chăm sóc đầu ca/tuần; theo dõi số khách được liên hệ và số khách quay lại.
    """)
    st.markdown("#### 6. Chỉ số nên theo dõi sau chiến dịch")
    st.write("Contact rate · Return rate trong 30/60/90 ngày · Repeat purchase rate · Incremental revenue · Voucher redemption · Cost per reactivated member · Tỷ lệ khách từ chối nhận tin.")
