import os
import requests
import streamlit as st

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="GAN-Based Multiclass Ag-IoT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# MODERN IDS DEPLOYMENT TOOL STYLING
# ==================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Layout & Soft Cool-Gray Background #F4F7FA */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #F4F7FA !important;
        color: #0F172A !important;
    }

    /* Hide Streamlit Chrome */
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stDeployButton"] {
        display: none !important;
    }
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Compact Application Container */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.75rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }

    /* Header Container & Typography */
    .header-container {
        text-align: center;
        margin-bottom: 1.75rem;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.25;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
    }

    .sub-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #334155;
        letter-spacing: -0.015em;
        margin: 0 0 0.45rem 0;
    }

    .header-desc {
        font-size: 0.92rem;
        color: #64748B;
        font-weight: 500;
        margin: 0;
        line-height: 1.45;
    }

    /* 3 Main White Cards (#E2E8F0 border, rounded, subtle shadows) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.35rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 6px 16px -2px rgba(15, 23, 42, 0.04) !important;
        min-height: 460px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 10px 24px -4px rgba(15, 23, 42, 0.06) !important;
    }

    /* Section Headings & Badges */
    .col-header {
        margin-bottom: 1.25rem;
        min-height: 84px;
    }

    .col-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.45rem;
    }

    .badge-step {
        color: #2563EB;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
    }

    .badge-download {
        color: #16A34A;
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
    }

    .col-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
    }

    .col-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin: 0;
        font-weight: 500;
        line-height: 1.35;
    }

    /* Button Layout & Spacing */
    div[data-testid="stButton"] {
        margin-bottom: 0.45rem !important;
    }

    /* Selection Buttons Sizing (54-58px height) */
    div[data-testid="stButton"] button {
        height: 56px !important;
        min-height: 56px !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Unselected Buttons: white background, #E2E8F0 border, dark text */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #E2E8F0 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }

    div[data-testid="stButton"] button[kind="secondary"] p,
    div[data-testid="stButton"] button[kind="secondary"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #CBD5E1 !important;
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.05) !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover p,
    div[data-testid="stButton"] button[kind="secondary"]:hover span {
        color: #0F172A !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:active {
        transform: translateY(0) !important;
    }

    /* Selected Buttons: green #16A34A, white text, checkmark */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #16A34A !important;
        background: #16A34A !important;
        color: #FFFFFF !important;
        border: 1.5px solid #15803D !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.22) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stButton"] button[kind="primary"] p,
    div[data-testid="stButton"] button[kind="primary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #15803D !important;
        background: #15803D !important;
        border-color: #166534 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(22, 163, 74, 0.32) !important;
        transform: translateY(-1.5px) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover p,
    div[data-testid="stButton"] button[kind="primary"]:hover span {
        color: #FFFFFF !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* Selected Configuration Card in Step 03 */
    .summary-box {
        background: #F8FAFC;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.15rem 1.15rem;
        margin-bottom: 1.25rem;
    }

    .summary-item {
        margin-bottom: 0.95rem;
    }

    .summary-item:last-child {
        margin-bottom: 0;
    }

    .summary-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
    }

    .summary-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }

    .summary-code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: #0F172A;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        padding: 0.45rem 0.65rem;
        border-radius: 8px;
        word-break: break-all;
    }

    /* Download Button Styling (green #16A34A, 56px height) */
    div[data-testid="stDownloadButton"] {
        width: 100% !important;
    }

    div[data-testid="stDownloadButton"] button {
        background-color: #16A34A !important;
        background: #16A34A !important;
        color: #FFFFFF !important;
        border: 1.5px solid #15803D !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        height: 56px !important;
        min-height: 56px !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25) !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
    }

    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background-color: #15803D !important;
        background: #15803D !important;
        border-color: #166534 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(22, 163, 74, 0.35) !important;
        transform: translateY(-1.5px) !important;
    }

    div[data-testid="stDownloadButton"] button:hover p,
    div[data-testid="stDownloadButton"] button:hover span {
        color: #FFFFFF !important;
    }

    div[data-testid="stDownloadButton"] button:active {
        transform: translateY(0) !important;
    }

    /* Empty State Box in Step 03 */
    .empty-state-box {
        background: #F8FAFC;
        border: 1.5px dashed #CBD5E1;
        border-radius: 12px;
        padding: 2.25rem 1.25rem;
        text-align: center;
        color: #64748B;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 260px;
    }

    .empty-icon {
        margin-bottom: 0.75rem;
        color: #94A3B8;
    }

    .empty-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.35rem;
    }

    .empty-desc {
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.45;
        max-width: 250px;
    }

    /* Responsive Design */
    @media (max-width: 992px) {
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            min-height: auto !important;
            margin-bottom: 1.25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# DEPLOYMENT ENGINES REGISTRY (10 COMBINATIONS)
# ==================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

DEPLOYMENT_ENGINES = {
    "CTGAN": {
        "Decision Tree": "DT_CTGAN_Deployment_Engine.zip",
        "MLP": "MLP_CTGAN_Deployment_Engine.zip",
        "Random Forest": "RF_CTGAN_Deployment_Engine.zip",
        "XGBoost": "XGB_CTGAN_Deployment_Engine.zip",
        "TabNet": "TabNet_CTGAN_Deployment_Engine.zip",
    },
    "CWGAN": {
        "Decision Tree": "DT_CWGAN_Deployment_Engine.zip",
        "MLP": "MLP_CWGAN_Deployment_Engine.zip",
        "Random Forest": "RF_CWGAN_Deployment_Engine.zip",
        "XGBoost": "XGB_CWGAN_Deployment_Engine.zip",
        "TabNet": "TabNet_CWGAN_Deployment_Engine.zip",
    },
}

# ==================================================
# STATE INITIALIZATION
# ==================================================
if "selected_gan" not in st.session_state:
    st.session_state["selected_gan"] = None

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="header-container">
    <h1 class="main-title">GAN-Based Data Augmentation</h1>
    <div class="sub-title">For Multiclass Intrusion Detection in Ag-IoT</div>
    <div class="header-desc">Select an augmentation method and IDS model to download the matching deployment engine.</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# THREE BALANCED COLUMNS
# ==================================================
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

# --------------------------------------------------
# CARD 1: STEP 01 — GAN AUGMENTATION
# --------------------------------------------------
with col1:
    with st.container(border=True):
        st.markdown("""
        <div class="col-header">
            <span class="col-badge badge-step">STEP 01</span>
            <div class="col-title">GAN Augmentation</div>
            <div class="col-desc">Select the augmentation method.</div>
        </div>
        """, unsafe_allow_html=True)

        is_ctgan = (st.session_state["selected_gan"] == "CTGAN")
        if st.button("✓ CTGAN" if is_ctgan else "CTGAN", type="primary" if is_ctgan else "secondary", use_container_width=True, key="btn_ctgan"):
            st.session_state["selected_gan"] = "CTGAN"
            st.rerun()

        is_cwgan = (st.session_state["selected_gan"] == "CWGAN")
        if st.button("✓ CWGAN" if is_cwgan else "CWGAN", type="primary" if is_cwgan else "secondary", use_container_width=True, key="btn_cwgan"):
            st.session_state["selected_gan"] = "CWGAN"
            st.rerun()

# --------------------------------------------------
# CARD 2: STEP 02 — IDS MODEL
# --------------------------------------------------
with col2:
    with st.container(border=True):
        st.markdown("""
        <div class="col-header">
            <span class="col-badge badge-step">STEP 02</span>
            <div class="col-title">IDS Model</div>
            <div class="col-desc">Select the model you want to download.</div>
        </div>
        """, unsafe_allow_html=True)

        models = ["Decision Tree", "MLP", "Random Forest", "XGBoost", "TabNet"]
        for m in models:
            is_active = (st.session_state["selected_model"] == m)
            label = f"✓ {m}" if is_active else m
            if st.button(label, type="primary" if is_active else "secondary", use_container_width=True, key=f"btn_{m.replace(' ', '_')}"):
                st.session_state["selected_model"] = m
                st.rerun()

# --------------------------------------------------
# CARD 3: DOWNLOAD — SELECTED MODEL
# --------------------------------------------------
with col3:
    with st.container(border=True):
        st.markdown("""
        <div class="col-header">
            <span class="col-badge badge-download">DOWNLOAD</span>
            <div class="col-title">Selected Model</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["selected_gan"] is not None and st.session_state["selected_model"] is not None:
            gan = st.session_state["selected_gan"]
            ids_model = st.session_state["selected_model"]

            target_filename = DEPLOYMENT_ENGINES.get(gan, {}).get(ids_model, f"{gan}_{ids_model}_Deployment_Engine.zip")

            is_available = False
            backend_error_msg = None

            try:
                check_resp = requests.head(
                    f"{BACKEND_URL}/download-model",
                    params={"gan": gan, "model": ids_model},
                    timeout=5
                )
                if check_resp.status_code == 200:
                    is_available = True
                elif check_resp.status_code == 404:
                    backend_error_msg = f"Deployment engine '{target_filename}' was not found on backend (HTTP 404)."
                else:
                    backend_error_msg = f"Backend returned HTTP status {check_resp.status_code}."
            except requests.exceptions.RequestException:
                backend_error_msg = f"Cannot connect to backend server at {BACKEND_URL}. Please ensure FastAPI is running."

            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-item">
                    <div class="summary-title">Selected Configuration</div>
                    <div class="summary-value">{gan} + {ids_model}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-title">Deployment Engine</div>
                    <div class="summary-code">{target_filename}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_available:
                def fetch_deployment_engine_zip() -> bytes:
                    dl_resp = requests.get(
                        f"{BACKEND_URL}/download-model",
                        params={"gan": gan, "model": ids_model},
                        timeout=300
                    )
                    if dl_resp.status_code == 200:
                        return dl_resp.content
                    raise RuntimeError(f"Download failed with HTTP {dl_resp.status_code}")

                st.download_button(
                    label="DOWNLOAD DEPLOYMENT ENGINE",
                    data=fetch_deployment_engine_zip,
                    file_name=target_filename,
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                    key="btn_download_engine"
                )
            else:
                st.error(backend_error_msg or "Deployment engine is unavailable.")

        else:
            if st.session_state["selected_gan"] is None and st.session_state["selected_model"] is None:
                status_msg = "Select an augmentation method and IDS model to download the matching deployment engine."
            elif st.session_state["selected_gan"] is None:
                status_msg = "Select an augmentation method in Step 01 to complete configuration."
            else:
                status_msg = "Select an IDS model in Step 02 to complete configuration."

            st.markdown(f"""
            <div class="empty-state-box">
                <div class="empty-icon">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                </div>
                <div class="empty-title">Awaiting Selection</div>
                <div class="empty-desc">{status_msg}</div>
            </div>
            """, unsafe_allow_html=True)
