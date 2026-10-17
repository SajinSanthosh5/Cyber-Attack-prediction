import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# Load Pickle File
# -------------------------------
with open("pickle.pkl", "rb") as f:
    data = pickle.load(f)

# Extract objects from pickle
preprocessor = data.get("preprocesser", None)
model = data.get("model", None)
features = data.get("feature", None)
label_encoder = data.get("label_encoder", None)
class_mapping = data.get("class_mapping", None)  # optional custom reverse map

# Basic validation
if model is None or features is None:
    st.error("Model or feature list not found in pickle.pkl. Make sure 'model' and 'feature' keys exist.")
    st.stop()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Cyber Attack Solution Predictor", page_icon="🔐")
st.title("🔐 Cyber Attack Solution Predictor")
st.write("Select the values for each feature to predict the best (text) solution.")

# Example options dictionary (extend to suit your features)
options_dict = {
    "Category": [
        "mobile security", "ai agents llm exploit", "ai/ml security",
        "cloud security", "network security", "web application security",
        "malware threat", "email message protocol exploit",
        "wireless attack advance", "quantum cryptography", "iot & embed devices",
    ],
    "Attack Type": [
        "SQL Injection (SQLi)", "SQL Injection", "RF Interference Simulation",
        "Filesystem Fault Simulation", "Constellation Relay Spoofing"
    ],
    "Tools Used": [
        "Browser, Burp Suite, SQLMap", "SQLMap, Burp Suite, Havij, Browser Developer Tools",
        "Kubernetes, Cyber Range Kits", "Ghidra, Custom Firmware Scanner",
    ],
    "Target Type": [
        "Web Login Portals (e.g., banking, admin dashboards, e-commerce)",
        "Web Applications, Login Pages, Search Forms"
    ],
    "MITRE Technique": ["T1078", "T1190", "T1505.003", "T0895"],
    "Impact": ["Data theft", "Privilege escalation", "Information disclosure"],
    "Detection Method": ["WAF alerts", "Log analysis", "Static binary analysis"],
    "Source": ["OWASP, MITRE ATT&CK, DVWA", "Internal Tools", "National Lab"]
}

# -------------------------------
# Collect User Inputs
# -------------------------------
user_inputs = {}
for col in features:
    if col in options_dict:
        user_inputs[col] = st.selectbox(f"{col}", options_dict[col])
    else:
        # use text_input but prefill empty string
        user_inputs[col] = st.text_input(f"{col}", value="")

# -------------------------------
# Helper: attempt to decode prediction to text label
# -------------------------------
def decode_prediction(y_pred, label_encoder=None, model=None, class_mapping=None):
    # 1) None check
    if y_pred is None:
        return None, "raw_is_none"

    # 2) If already a string, assume that's the label
    if isinstance(y_pred, str):
        return y_pred, "already_str"

    # Ensure we have a scalar numeric or something convertible
    try:
        # if numpy scalar or array element
        if isinstance(y_pred, (np.generic, np.ndarray)):
            y_pred = np.asarray(y_pred).item()
    except Exception:
        pass

    # 3) Try label_encoder (sklearn LabelEncoder or similar)
    if label_encoder is not None:
        try:
            # many LabelEncoders expect array-like of ints
            decoded = label_encoder.inverse_transform([int(y_pred)])[0]
            return decoded, "label_encoder_inverse_transform"
        except Exception:
            # maybe label_encoder stores classes_ or is a mapping
            try:
                if hasattr(label_encoder, "classes_"):
                    cl = getattr(label_encoder, "classes_")
                    idx = int(y_pred)
                    if 0 <= idx < len(cl):
                        return cl[idx], "label_encoder_classes_"
            except Exception:
                pass
            # try if it's a dict-like mapping (value->label)
            if isinstance(label_encoder, dict):
                # try reverse lookup: value->key? We'll try to match value
                try:
                    # If label_encoder was saved as {label: index}, reverse it:
                    rev = {v: k for k, v in label_encoder.items()}
                    if y_pred in rev:
                        return rev[y_pred], "label_encoder_rev_dict"
                except Exception:
                    pass
            # give up on label_encoder route
    # 4) Try provided class_mapping (explicit mapping index->label)
    if class_mapping is not None:
        try:
            # if mapping is index->label
            if y_pred in class_mapping:
                return class_mapping[y_pred], "class_mapping_index"
            # maybe mapping was label->index, reverse it
            if isinstance(class_mapping, dict):
                rev = {v: k for k, v in class_mapping.items()}
                if y_pred in rev:
                    return rev[y_pred], "class_mapping_reversed"
        except Exception:
            pass

    # 5) Try estimator.classes_ (works if model stored classifier)
    try:
        # If pipeline, try to locate an estimator with classes_
        candidate = model
        if hasattr(model, "named_steps"):  # sklearn Pipeline
            # try to find last estimator with classes_
            for name, step in reversed(list(model.named_steps.items())):
                if hasattr(step, "classes_"):
                    candidate = step
                    break
        if hasattr(candidate, "classes_"):
            classes = getattr(candidate, "classes_")
            idx = int(y_pred)
            if 0 <= idx < len(classes):
                return classes[idx], "model_classes_"
    except Exception:
        pass

    # 6) If y_pred numeric but small, try to string it
    try:
        return str(y_pred), "fallback_str"
    except Exception:
        return None, "fallback_fail"

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Solution"):
    try:
        # Convert to DataFrame with expected column order
        input_df = pd.DataFrame([user_inputs])[features]

        # Preprocess if provided
        if preprocessor is not None:
            X_processed = preprocessor.transform(input_df)
            raw_pred = model.predict(X_processed)
        else:
            raw_pred = model.predict(input_df)

        st.write("🔍 Raw prediction from model (full):", raw_pred)
        # If array-like, show first element separately
        if hasattr(raw_pred, "__len__"):
            first = raw_pred[0] if len(raw_pred) > 0 else None
        else:
            first = raw_pred

        st.write("🔎 First element (scalar) from model:", first, "   (type:", type(first), ")")

        # decode
        decoded, method = decode_prediction(first, label_encoder=label_encoder, model=model, class_mapping=class_mapping)

        if decoded is None:
            st.warning("⚠️ Could not decode the model output to a text label.")
            st.info(
                "Debug hints:\n"
                "- Raw model output shown above.\n"
                "- If the model returns a numeric index, you must save a LabelEncoder or class mapping when training.\n"
                "- When training, save something like: {'model': model, 'preprocesser': preprocessor, 'feature': features, 'label_encoder': label_encoder}\n"
                "- Example mapping keys found in pickle: " + ", ".join([k for k in data.keys()])
            )
            st.error(f"Raw value returned by model: {first} (type: {type(first)})")
        else:
            st.success(f"✅ Recommended Solution: {decoded}")
            st.caption(f"(decoded using {method})")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.exception(e)
