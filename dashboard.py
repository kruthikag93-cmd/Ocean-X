import streamlit as st
import json
import pandas as pd
from PIL import Image
import subprocess
import sys
import os
import uuid
import hashlib
import shutil


# ============================================================
# OCEAN X
# ============================================================

st.set_page_config(
    page_title="Ocean X | Marine Habitat Protection",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PIPELINE = os.path.join(
    BASE_DIR,
    "final_pipeline.py"
)

RESULT_JSON = os.path.join(
    BASE_DIR,
    "ocean_x_results.json"
)

RESULT_CSV = os.path.join(
    BASE_DIR,
    "ocean_x_results.csv"
)

RESULT_IMAGE = os.path.join(
    BASE_DIR,
    "ocean_x_final_detection.png"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "dashboard_uploads"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "dashboard_results"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "current_result_image" not in st.session_state:
    st.session_state.current_result_image = None


# ============================================================
# SIMPLE CLEAN UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7fbfc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #063b4a;
        margin-bottom: 0;
    }

    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #087f8c;
        margin-top: 3px;
    }

    .main-description {
        color: #61777c;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    .ready-box {
        min-height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;

        border: 2px dashed #a9cfd4;
        border-radius: 14px;

        background: #f8fcfd;
        color: #58757b;
    }

    .ready-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #075985;
        margin-bottom: 8px;
    }

    .status-positive {
        background: #fff7e8;
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 16px;
        color: #713f12;
    }

    .status-negative {
        background: #ecfdf3;
        border-left: 5px solid #16a34a;
        border-radius: 10px;
        padding: 16px;
        color: #14532d;
    }

    .status-title {
        font-weight: 750;
        font-size: 1.15rem;
    }

    .status-text {
        margin-top: 5px;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: #71878c;
        padding-top: 25px;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — KEEP IT SIMPLE
# ============================================================

st.sidebar.title("🌊 Ocean X")

st.sidebar.write(
    "**Marine Habitat Protection**"
)

st.sidebar.write(
    "AI-powered analysis of side-scan sonar imagery "
    "for identifying potential man-made underwater targets."
)

st.sidebar.divider()

st.sidebar.success(
    "Ocean X V3"
)

st.sidebar.info(
    "SIH26057"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Ocean X</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-Powered Marine Debris Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Marine Habitat Protection • Side-Scan Sonar Analysis • SIH26057'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SHORT INTRO
# ============================================================

st.info(
    "🐠 **Marine Habitat Protection** — "
    "Ocean X analyzes side-scan sonar imagery to identify "
    "potential man-made underwater targets that may require "
    "further inspection."
)


# ============================================================
# UPLOAD
# ============================================================

st.header("📡 Sonar Image")

uploaded_file = st.file_uploader(
    "Upload a side-scan sonar image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a sonar image to begin Ocean X analysis."
    )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    # --------------------------------------------------------
    # CREATE CONTENT FINGERPRINT
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    file_id = hashlib.md5(
        file_bytes
    ).hexdigest()


    # --------------------------------------------------------
    # NEW IMAGE?
    # --------------------------------------------------------

    if (
        st.session_state.current_file_id
        != file_id
    ):

        st.session_state.current_file_id = file_id

        st.session_state.analysis_complete = False

        st.session_state.analysis_data = None

        st.session_state.current_result_image = None


        # Delete old pipeline result files

        for old_file in [
            RESULT_JSON,
            RESULT_CSV,
            RESULT_IMAGE
        ]:

            if os.path.exists(
                old_file
            ):

                try:
                    os.remove(
                        old_file
                    )

                except Exception:
                    pass


    # --------------------------------------------------------
    # SAVE CURRENT UPLOAD
    # --------------------------------------------------------

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    uploaded_path = os.path.join(
        UPLOAD_DIR,
        "sonar_" + file_id + extension
    )

    with open(
        uploaded_path,
        "wb"
    ) as f:

        f.write(
            file_bytes
        )


    image = Image.open(
        uploaded_path
    )


    # ========================================================
    # ORIGINAL + AI ANALYSIS
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------------------------
    # ORIGINAL SONAR
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "📡 Original Sonar"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            uploaded_file.name
        )


    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "🤖 AI Analysis"
        )


        # ====================================================
        # BEFORE ANALYSIS
        # ====================================================

        if not st.session_state.analysis_complete:

            st.html(
                """
                <div class="ready-box">

                    <div>

                        <div style="font-size: 2.5rem;">
                            🔍
                        </div>

                        <div class="ready-title">
                            Ready for Analysis
                        </div>

                        <div>
                            Click <b>Analyze Sonar Image</b>
                            to run Ocean X V3.
                        </div>

                    </div>

                </div>
                """,
                
            )


        # ====================================================
        # AFTER ANALYSIS
        # ====================================================

        else:

            current_result = (
                st.session_state.current_result_image
            )

            if (
                current_result
                and
                os.path.exists(current_result)
            ):

                st.image(
                    current_result,
                    use_container_width=True
                )

                st.success(
                    "✅ Analysis completed"
                )

                st.caption(
                    "Ocean X AI analysis for this uploaded image"
                )

            else:

                st.warning(
                    "Analysis completed, but the result "
                    "visualization is unavailable."
                )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.write("")

    analyze = st.button(
        "🚀 Analyze Sonar Image",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN PIPELINE
    # ========================================================

    if analyze:

        # ----------------------------------------------------
        # REMOVE OLD PIPELINE OUTPUT
        # ----------------------------------------------------

        for old_file in [
            RESULT_JSON,
            RESULT_CSV,
            RESULT_IMAGE
        ]:

            if os.path.exists(
                old_file
            ):

                try:
                    os.remove(
                        old_file
                    )

                except Exception:
                    pass


        # ----------------------------------------------------
        # RESET CURRENT RESULT
        # ----------------------------------------------------

        st.session_state.analysis_complete = False

        st.session_state.analysis_data = None

        st.session_state.current_result_image = None


        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        env = os.environ.copy()

        env[
            "OCEAN_X_IMAGE_PATH"
        ] = uploaded_path


        # ----------------------------------------------------
        # RUN AI
        # ----------------------------------------------------

        with st.spinner(
            "🌊 Ocean X is analyzing the sonar image..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        PIPELINE
                    ],
                    cwd=BASE_DIR,
                    env=env,
                    capture_output=True,
                    text=True
                )


                # ============================================
                # ERROR
                # ============================================

                if result.returncode != 0:

                    st.error(
                        "❌ Ocean X analysis failed."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            result.stderr
                        )


                # ============================================
                # SUCCESS
                # ============================================

                else:

                    # ----------------------------------------
                    # LOAD JSON
                    # ----------------------------------------

                    if os.path.exists(
                        RESULT_JSON
                    ):

                        with open(
                            RESULT_JSON,
                            "r"
                        ) as f:

                            data = json.load(
                                f
                            )


                        # ------------------------------------
                        # CREATE UNIQUE RESULT IMAGE
                        # ------------------------------------

                        unique_result = os.path.join(
                            RESULT_DIR,
                            "analysis_"
                            + file_id
                            + ".png"
                        )


                        if os.path.exists(
                            RESULT_IMAGE
                        ):

                            shutil.copy2(
                                RESULT_IMAGE,
                                unique_result
                            )


                        # ------------------------------------
                        # STORE CURRENT RESULT ONLY
                        # ------------------------------------

                        st.session_state.analysis_data = data

                        st.session_state.current_result_image = (
                            unique_result
                            if os.path.exists(
                                unique_result
                            )
                            else None
                        )

                        st.session_state.analysis_complete = True


                        # ------------------------------------
                        # REFRESH PAGE
                        # ------------------------------------

                        st.rerun()


                    else:

                        st.error(
                            "Pipeline completed, but "
                            "no JSON result was generated."
                        )


            except Exception as e:

                st.error(
                    "❌ Unable to run Ocean X pipeline."
                )

                st.exception(
                    e
                )


    # ========================================================
    # RESULTS
    # ========================================================

    if st.session_state.analysis_complete:

        data = st.session_state.analysis_data

        detections = data.get(
            "detections",
            []
        )

        width = data.get(
            "image_width",
            image.width
        )

        height = data.get(
            "image_height",
            image.height
        )


        # ====================================================
        # STATUS
        # ====================================================

        st.divider()

        st.header(
            "🔎 Detection Status"
        )


        if detections:

            st.html(
                f"""
                <div class="status-positive">

                    <div class="status-title">
                        🟠 Potential Man-Made Target Detected
                    </div>

                    <div class="status-text">

                        Ocean X identified
                        <b>{len(detections)}</b>
                        candidate target region(s)
                        requiring further inspection.

                        <br><br>

                        These are candidate AI detections
                        and are not confirmed classifications
                        of marine debris.

                    </div>

                </div>
                """,
                
            )

        else:

            st.html(
                """
                <div class="status-negative">

                    <div class="status-title">
                        🟢 No Potential Marine Debris Detected
                    </div>

                    <div class="status-text">

                        No candidate target regions passed
                        the current Ocean X detection filters.

                    </div>

                </div>
                """,
                
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "🎯 Detection Summary"
        )


        if detections:

            highest = max(
                d["max_confidence"]
                for d in detections
            )

        else:

            highest = 0


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Candidate Regions",
                len(detections)
            )


        with c2:

            st.metric(
                "Highest Confidence",
                f"{highest * 100:.1f}%"
            )


        with c3:

            st.metric(
                "Image Width",
                width
            )


        with c4:

            st.metric(
                "Image Height",
                height
            )


        # ====================================================
        # AI VISUALIZATION
        # ====================================================

        st.divider()

        st.header(
            "🛰️ Ocean X AI Visualization"
        )


        current_result = (
            st.session_state.current_result_image
        )


        if (
            current_result
            and
            os.path.exists(current_result)
        ):

            st.image(
                current_result,
                use_container_width=True,
                caption=(
                    "Original Sonar • "
                    "AI Probability Map • "
                    "Candidate Detection"
                )
            )


        # ====================================================
        # DETECTION TABLE
        # ====================================================

        st.divider()

        st.header(
            "📊 Candidate Target Regions"
        )


        if detections:

            rows = []


            for d in detections:

                rows.append(
                    {
                        "Target":
                            d["target_id"],

                        "Area":
                            d["area"],

                        "Mean Confidence":
                            (
                                f"{d['mean_confidence'] * 100:.1f}%"
                            ),

                        "Max Confidence":
                            (
                                f"{d['max_confidence'] * 100:.1f}%"
                            ),

                        "Bounding Box":
                            (
                                f"({d['x_min']}, "
                                f"{d['y_min']}) → "
                                f"({d['x_max']}, "
                                f"{d['y_max']})"
                            )
                    }
                )


            df = pd.DataFrame(
                rows
            )


            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "🟢 No candidate target regions were identified."
            )


        # ====================================================
        # REPORTS
        # ====================================================

        st.divider()

        st.header(
            "📄 Analysis Reports"
        )


        c1, c2 = st.columns(2)


        with c1:

            if os.path.exists(
                RESULT_CSV
            ):

                with open(
                    RESULT_CSV,
                    "rb"
                ) as f:

                    st.download_button(
                        "⬇️ Download CSV",
                        f,
                        file_name="ocean_x_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


        with c2:

            if os.path.exists(
                RESULT_JSON
            ):

                with open(
                    RESULT_JSON,
                    "rb"
                ) as f:

                    st.download_button(
                        "⬇️ Download JSON",
                        f,
                        file_name="ocean_x_results.json",
                        mime="application/json",
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        🌊 Ocean X • SIH26057 • Marine Habitat Protection

    </div>
    """,
    unsafe_allow_html=True
)