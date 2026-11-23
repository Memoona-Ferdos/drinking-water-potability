# ==========================================================
# DRINKING WATER POTABILITY ASSESSMENT AND SAFETY PREDICTOR
# Main Streamlit Application
# ==========================================================
# RUN THIS FILE WITH:   streamlit run app/app.py
# (Do NOT run it with "python app.py" or the VS Code Run button.)
# ==========================================================

import sys

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==========================================================
# 0. LAUNCH GUARD & AUTO-BOOTSTRAPPER
# ==========================================================
# If this file is started as a normal Python script (e.g. from
# VS Code's Run button or `python app/app.py`), automatically
# bootstrap the Streamlit server using Python. This avoids
# "missing ScriptRunContext" warnings and bypasses Windows
# Application Control blocking on streamlit.exe.


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return True  # If detection fails, do not block the app.


if not running_inside_streamlit():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================
# Walk upward from this file until the folder containing
# models/best_model.pkl is found. Works at any nesting depth.


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "models" / "best_model.pkl").exists():
            return folder
    # Nothing found: fall back so the error message shows a useful path.
    return start


BASE_DIR = find_project_root(Path(__file__).resolve().parent)

DATA_PATH = BASE_DIR / "data" / "water_potability.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

REPORT_PATH = BASE_DIR / "reports"

FIGURE_PATH = REPORT_PATH / "figures"

COMPARISON_PATH = REPORT_PATH / "model_comparison.csv"


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f8fb;
}

[data-testid="stSidebar"] {
    background-color: #eaf4f8;
}

h1 {
    color: #075985;
}

h2, h3 {
    color: #0369a1;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dbeafe;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3b. STREAMLIT VERSION COMPATIBILITY HELPERS
# ==========================================================
# Newer Streamlit versions replace use_container_width=True with
# width="stretch". These helpers try the new form first and fall
# back to the old one, so the app runs on either version.


def show_dataframe(data):
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


def show_image(path):
    try:
        st.image(str(path), width="stretch")
    except TypeError:
        st.image(str(path), use_container_width=True)


def stretch_button(label):
    try:
        return st.form_submit_button(label, width="stretch")
    except TypeError:
        return st.form_submit_button(label, use_container_width=True)


# ==========================================================
# 4. LOAD DATASET AND TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    data.columns = data.columns.str.strip()

    return data


try:

    model = load_model()

    df = load_dataset()

except Exception as error:

    st.error(f"Application loading error: {error}")

    st.stop()


# ==========================================================
# 5. FEATURE CONFIGURATION
# ==========================================================

features = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# Confirm that the model uses the expected features. Some saved
# models do not store feature names, so check safely first.
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None and list(model_features) != features:

    st.error(
        "Feature order mismatch. "
        "Check the feature columns used during model training.\n\n"
        f"Model expects: {list(model_features)}"
    )

    st.stop()

if not all(feature in df.columns for feature in features):

    st.error("One or more expected dataset columns are missing.")

    st.stop()

if "Potability" not in df.columns:

    st.error("Target column Potability is missing.")

    st.stop()


feature_labels = {

    "ph": "pH Level",

    "Hardness": "Hardness",

    "Solids": "Total Dissolved Solids",

    "Chloramines": "Chloramines",

    "Sulfate": "Sulfate",

    "Conductivity": "Conductivity",

    "Organic_carbon": "Organic Carbon",

    "Trihalomethanes": "Trihalomethanes",

    "Turbidity": "Turbidity"
}


feature_units = {

    "ph": "",

    "Hardness": "mg/L",

    "Solids": "ppm",

    "Chloramines": "mg/L",

    "Sulfate": "mg/L",

    "Conductivity": "μS/cm",

    "Organic_carbon": "mg/L",

    "Trihalomethanes": "μg/L",

    "Turbidity": "NTU"
}


# ==========================================================
# 6. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("💧 Water Predictor")

st.sidebar.caption(
    "AI & Mobile Robotics Project"
)

page = st.sidebar.radio(

    "Navigate",

    [
        "Home",
        "Water Potability Predictor",
        "🤖 Mobile Robot Mission Control",
        "Exploratory Data Analysis",
        "Model Performance"
    ]

)

st.sidebar.divider()

st.sidebar.info(
    "🤖 **AI & Mobile Robotics System**\n\n"
    "- Edge Classifier: Best saved model\n"
    "- Onboard Compute: Jetson / RPi (design)\n"
    "- GCS Dashboard: Streamlit\n"
    "- Headless mode: command-line script (optional)"
)


# ==========================================================
# 7. HOME PAGE
# ==========================================================

if page == "Home":

    st.title("💧 Drinking Water Potability Predictor")

    st.subheader(
        "AI & Autonomous Mobile Robotics Environmental Screening System"
    )

    st.write(
        """
        Welcome to the **Drinking Water Potability Assessment & Autonomous Robotic Screening System**.

        This system combines **Machine Learning classification** with **Mobile Robotics Architecture**
        for autonomous environmental water monitoring. An aquatic mobile robot (Unmanned Surface Vehicle - USV)
        navigates water bodies, samples multi-parameter water quality sensors, and uses an onboard
        edge AI classifier to make sampling and alerting decisions.
        """
    )

    st.divider()

    total_samples = len(df)

    total_features = len(features)

    missing_values = int(df[features].isnull().sum().sum())

    potable_samples = int(
        (df["Potability"] == 1).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Samples",
        total_samples
    )

    col2.metric(
        "Input Features",
        total_features
    )

    col3.metric(
        "Potable Class Samples",
        potable_samples
    )

    col4.metric(
        "Missing Measurements",
        missing_values
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze water-quality measurements.
    - Perform exploratory data analysis.
    - Handle missing values using median imputation.
    - Train Decision Tree and Random Forest classifiers.
    - Compare model performance.
    - Predict the dataset potability class for new measurements.
    - Demonstrate how the classifier supports a mobile-robot sampling mission (simulation).
    """)

    st.subheader("Machine Learning Workflow")

    st.info(
        "Dataset → Preprocessing → Train/Test Split → "
        "Machine Learning → Evaluation → Prediction"
    )

    st.subheader("Input Water Quality Parameters")

    st.write(", ".join(feature_labels.values()))

    st.warning(
        "This is an educational machine-learning screening tool. "
        "Its predictions do not establish whether water is safe "
        "to drink. Actual drinking-water safety requires "
        "appropriate laboratory testing and regulatory assessment."
    )


# ==========================================================
# 8. WATER POTABILITY PREDICTOR
# ==========================================================

elif page == "Water Potability Predictor":

    st.title("💧 Water Potability Prediction")

    st.write(
        "Enter the nine water-quality measurements below."
    )

    st.info(
        "Enter measurements in the units shown. "
        "The default values are dataset medians for convenience."
    )

    # Use training-dataset medians as convenient defaults.
    medians = df[features].median()

    with st.form("water_prediction_form"):

        st.subheader("Enter Water Quality Measurements")

        col1, col2, col3 = st.columns(3)

        user_values = {}

        columns = [col1, col2, col3]

        for index, feature in enumerate(features):

            current_column = columns[index % 3]

            with current_column:

                label = feature_labels[feature]

                unit = feature_units[feature]

                if unit:
                    label = f"{label} ({unit})"

                user_values[feature] = st.number_input(

                    label,

                    value=float(medians[feature]),

                    format="%.4f",

                    key=f"input_{feature}"

                )

        submitted = stretch_button("🔍 Predict Potability")

    if submitted:

        # Maintain exactly the same feature order as training.
        input_data = pd.DataFrame(
            [[user_values[feature] for feature in features]],
            columns=features
        )

        # Use the saved preprocessing + classifier pipeline.
        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.success(
                    "Predicted dataset class: POTABLE (1)"
                )

            else:

                st.error(
                    "Predicted dataset class: NON-POTABLE (0)"
                )

        with result_col2:

            st.metric(
                "Predicted Class",
                prediction
            )

        st.subheader("Model-Estimated Class Scores")

        probability_df = pd.DataFrame({

            "Class": [
                "Non-Potable (0)",
                "Potable (1)"
            ],

            "Estimated Score": probabilities

        })

        st.bar_chart(
            probability_df.set_index("Class")
        )

        st.caption(
            "These are the classifier's estimated class probabilities, "
            "not laboratory-confirmed safety probabilities."
        )

        st.subheader("Entered Water Measurements")

        show_dataframe(input_data)

        st.warning(
            "Do not use this prediction to decide whether water "
            "is safe to drink. A potable-class prediction is not "
            "a guarantee of safety. Obtain appropriate laboratory "
            "testing before making drinking-water decisions."
        )


# ==========================================================
# 8b. MOBILE ROBOT MISSION CONTROL & TELEMETRY (SIMULATION)
# ==========================================================

elif page == "🤖 Mobile Robot Mission Control":

    st.title("🤖 Autonomous Aquatic Mobile Robot - Mission Control")

    st.subheader(
        "Environmental Monitoring Surface Drone (USV) & Edge AI Ground Station"
    )

    st.info(
        "**Simulation notice:** No real robot is connected. Waypoints, "
        "battery, speed, signal and depth are illustrative demo values, "
        "and the water readings are rows taken from the training dataset "
        "to show how the classifier would drive robot decisions."
    )

    st.markdown("""
    In an **AI & Mobile Robotics** deployment, this water potability model would run directly
    on an **autonomous Unmanned Surface Vehicle (USV)** or aquatic rover navigating lakes,
    reservoirs, or treatment facilities.

    The robot samples environmental sensors, runs edge classification locally
    (no cloud latency), and decides whether to trigger a physical water sampling pump
    or flag a contamination zone.
    """)

    # Robot Architecture Expander
    with st.expander("🛠️ Mobile Robot Hardware & System Architecture", expanded=False):
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown("""
            **Proposed Hardware & Compute:**
            - **Chassis:** Twin-pontoon catamaran USV (differential thrust)
            - **Processor:** Raspberry Pi 5 / NVIDIA Jetson Nano running Linux
            - **Microcontroller:** ESP32 / STM32 for motor control & sensor I2C/SPI
            - **Comms Link:** 915 MHz LoRa Telemetry + 4G LTE Ground Station fallback
            """)
        with arch_col2:
            st.markdown("""
            **Proposed Sensor & Actuator Payload:**
            - **Sensors:** Multi-parameter probe (pH, EC conductivity, turbidity, TDS)
            - **Actuators:** 4x Servo-actuated sample collection bottles
            - **Sampling Pump:** 12V Peristaltic intake pump (50 mL/min)
            - **Navigation:** RTK-GPS + IMU (inertial waypoint navigation)
            """)

    st.divider()

    st.subheader("📍 Autonomous Patrol Mission & Telemetry")

    # Simulated patrol waypoints
    mission_waypoints = [
        {
            "id": 1,
            "name": "Reservoir North Inlet",
            "lat": 33.7490,
            "lon": -84.3880,
            "speed": 1.2,
            "battery": 98,
            "signal": -62,
            "depth": 4.5
        },
        {
            "id": 2,
            "name": "Industrial Runoff Confluence",
            "lat": 33.7498,
            "lon": -84.3865,
            "speed": 1.0,
            "battery": 94,
            "signal": -65,
            "depth": 3.8
        },
        {
            "id": 3,
            "name": "Central Aeration Basin",
            "lat": 33.7512,
            "lon": -84.3842,
            "speed": 1.4,
            "battery": 90,
            "signal": -60,
            "depth": 8.2
        },
        {
            "id": 4,
            "name": "Recreational Shore Zone",
            "lat": 33.7530,
            "lon": -84.3820,
            "speed": 1.1,
            "battery": 86,
            "signal": -68,
            "depth": 2.1
        },
        {
            "id": 5,
            "name": "Potable Water Extraction Intake",
            "lat": 33.7545,
            "lon": -84.3800,
            "speed": 0.8,
            "battery": 82,
            "signal": -58,
            "depth": 11.0
        }
    ]

    # Pre-select representative sensor rows from the dataset
    p0_samples = df[df["Potability"] == 0].head(2)
    p1_samples = df[df["Potability"] == 1].head(3)

    sample_rows = [
        p1_samples.iloc[0],
        p0_samples.iloc[0],
        p1_samples.iloc[1],
        p0_samples.iloc[1],
        p1_samples.iloc[2]
    ]

    selected_wp_name = st.selectbox(
        "Select Mobile Robot Waypoint / Target Station:",
        [f"Waypoint {wp['id']}: {wp['name']}" for wp in mission_waypoints]
    )

    wp_idx = int(
        selected_wp_name.split(":")[0].replace("Waypoint", "").strip()
    ) - 1

    current_wp = mission_waypoints[wp_idx]

    current_sensor_row = sample_rows[wp_idx]

    # Telemetry metrics row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    m_col1.metric(
        "Robot Battery",
        f"{current_wp['battery']}%",
        delta=f"-{100 - current_wp['battery']}%"
    )
    m_col2.metric("Cruising Speed", f"{current_wp['speed']} m/s")
    m_col3.metric("Water Depth", f"{current_wp['depth']} m")
    m_col4.metric("LoRa Signal", f"{current_wp['signal']} dBm")
    m_col5.metric("GPS Status", "3D RTK Fix", delta="12 Sats")

    # Map visualization
    map_df = pd.DataFrame([
        {"lat": wp["lat"], "lon": wp["lon"]} for wp in mission_waypoints
    ])

    st.caption("🗺️ **Patrol Route GPS Map** (simulated waypoints):")

    st.map(map_df, zoom=14)

    st.divider()

    # Sensor telemetry readings
    st.subheader(f"📊 Telemetry Stream from {current_wp['name']}")

    # Missing sensor values fall back to the dataset median
    current_features_dict = {
        feat: (
            float(current_sensor_row[feat])
            if pd.notnull(current_sensor_row[feat])
            else float(df[feat].median())
        )
        for feat in features
    }

    sensor_display_df = pd.DataFrame([
        {
            "Parameter": feature_labels[f],
            "Telemetry Value": f"{current_features_dict[f]:.3f}",
            "Unit": feature_units[f],
            "Dataset Median": f"{float(df[f].median()):.3f}"
        }
        for f in features
    ])

    show_dataframe(sensor_display_df)

    # Edge AI inference
    input_sample = pd.DataFrame([current_features_dict], columns=features)

    robot_pred = int(model.predict(input_sample)[0])

    robot_probs = model.predict_proba(input_sample)[0]

    st.subheader("⚡ Onboard Edge AI Inference & Robotic Decision")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:

        if robot_pred == 1:

            st.success("### ✅ CLASSIFICATION: POTABLE (1)")

            st.markdown(f"""
            - **Model confidence:** `{robot_probs[1]*100:.1f}%`
            - **Status:** Readings fall in the dataset's potable class.
            - **🤖 Planned robotic action:**
              - Engage the 12V peristaltic sampling pump for 10 seconds
              - Index the sampling carousel to bottle #{current_wp['id']}
              - Log the timestamped GPS waypoint as a candidate clean source
            """)

        else:

            st.error("### ⚠️ CLASSIFICATION: NON-POTABLE (0)")

            st.markdown(f"""
            - **Model confidence:** `{robot_probs[0]*100:.1f}%`
            - **Status:** Readings fall in the dataset's non-potable class.
            - **🤖 Planned robotic action:**
              - Skip bottle storage (preserve clean sample containers)
              - Flag the GPS coordinates as a possible hazard hotspot
              - Send a high-priority alert to the Ground Control Station
            """)

    with res_col2:

        st.write("**Classifier Confidence Distribution:**")

        prob_chart_df = pd.DataFrame({
            "Classification": ["Non-Potable (Hazard)", "Potable (Safe)"],
            "Probability": [robot_probs[0], robot_probs[1]]
        })

        st.bar_chart(prob_chart_df.set_index("Classification"))

    st.warning(
        "Educational simulation only. A 'potable' classification does not "
        "mean the water is safe to drink; confirm with laboratory testing."
    )

    st.divider()

    # Terminal vs app deployment
    st.subheader("💡 Deployment Design: Robot Edge vs. Ground Station")

    st.markdown("""
    A deployed system would use two complementary environments:

    1. **Headless robot edge (command line):**
       - A small Python script runs on the Raspberry Pi / Jetson with no display.
       - Low power use, no UI overhead, fast inference.
       - It loads `models/best_model.pkl` and classifies each sensor reading.

    2. **Ground Control Station (this Streamlit dashboard):**
       - Runs on the operator's laptop or base station.
       - Would receive telemetry from the robot over LoRa / Wi-Fi / 4G.
       - Provides maps, charts, statistical summaries and manual override controls.
    """)


# ==========================================================
# 9. EXPLORATORY DATA ANALYSIS
# ==========================================================

elif page == "Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the water-quality dataset through "
        "statistical summaries and visualizations."
    )

    st.subheader("Dataset Preview")

    show_dataframe(df.head(10))

    st.subheader("Statistical Summary")

    show_dataframe(df[features].describe())

    st.divider()

    # Target distribution
    st.subheader("Potability Class Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x="Potability",
        hue="Potability",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-Potable", "Potable"])

    ax.set_xlabel("Potability Class")
    ax.set_ylabel("Number of Samples")

    st.pyplot(fig)

    plt.close(fig)

    # Missing values
    st.subheader("Missing Value Analysis")

    missing = df[features].isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            x=missing.index,
            y=missing.values,
            ax=ax
        )

        ax.set_title("Missing Values by Feature")

        ax.set_xlabel("Features")

        ax.set_ylabel("Missing Count")

        plt.xticks(rotation=25)

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.success("No missing feature values found.")

    # Feature distributions
    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Select a feature",
        features
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=selected_feature,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_feature}"
    )

    st.pyplot(fig)

    plt.close(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Water Quality Feature Correlations")

    st.pyplot(fig)

    plt.close(fig)


# ==========================================================
# 10. MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write(
        "Compare the Decision Tree and Random Forest "
        "classifiers using the held-out test results."
    )

    if COMPARISON_PATH.exists():

        results = pd.read_csv(COMPARISON_PATH)

        if "Model" not in results.columns:

            st.error("Model column is missing from comparison CSV.")

            st.stop()

        st.subheader("Model Comparison Table")

        numeric_columns = results.select_dtypes(
            include="number"
        ).columns

        show_dataframe(
            results.style.format(
                {column: "{:.4f}" for column in numeric_columns}
            )
        )

        if "Macro F1" in results.columns:

            st.subheader("Macro F1 Comparison")

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.barplot(
                data=results,
                x="Model",
                y="Macro F1",
                hue="Model",
                legend=False,
                ax=ax
            )

            ax.set_ylim(0, 1)

            st.pyplot(fig)

            plt.close(fig)

    else:

        st.warning(
            "Model comparison CSV not found. "
            "Run the Session 3 notebook's save-results cell."
        )

    st.divider()

    st.subheader("Saved Evaluation Graphs")

    saved_figures = [

        (
            "Decision Tree Confusion Matrix",
            "08_decision_tree_confusion_matrix.png"
        ),

        (
            "Random Forest Confusion Matrix",
            "09_random_forest_confusion_matrix.png"
        ),

        (
            "ROC Curve Comparison",
            "10_roc_curve_comparison.png"
        ),

        (
            "Model Performance Comparison",
            "07_model_comparison.png"
        )

    ]

    for title, filename in saved_figures:

        image_path = FIGURE_PATH / filename

        if image_path.exists():

            st.write(f"**{title}**")

            show_image(image_path)

        else:

            st.info(
                f"{filename} is not available yet."
            )


# ==========================================================
# 11. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Drinking Water Potability Assessment and Safety Predictor | "
    "AI & Mobile Robotics Academic Project"
)