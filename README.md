# 🌊 Ocean X

## AI-Powered Automated Underwater Marine Debris and Anomaly Detection using Side-Scan Sonar Imagery

**Smart India Hackathon 2026 — SIH26057**  
**Ministry of Earth Sciences (MoES) | National Institute of Ocean Technology (NIOT)**

---

## 🌊 About Ocean X

Ocean X is an AI-assisted system designed to analyze **Side-Scan Sonar (SSS) imagery** and identify potential man-made underwater targets and anomalies.

The system aims to support faster underwater survey analysis and help identify areas that may pose risks to **marine life, aquatic habitats, and underwater ecosystems**.

Instead of relying entirely on manual inspection of large sonar surveys, Ocean X provides an automated analysis workflow that highlights suspicious regions for further investigation.

---

## 🎯 Problem

Underwater sonar surveys can produce large amounts of imagery containing:

- Marine debris
- Shipwrecks and man-made objects
- Acoustic shadows
- Natural seabed structures
- Speckle noise and imaging artifacts

Manual inspection can be time-consuming and difficult, particularly when sonar imagery varies in resolution and quality.

Ocean X explores an AI-assisted approach to reduce the amount of imagery that requires detailed manual inspection.

---

## 💡 Proposed Solution

Ocean X combines:

- Side-Scan Sonar imagery
- Image preprocessing
- AI-based segmentation
- Candidate-region detection
- Confidence estimation
- Visualization
- Structured JSON/CSV reporting
- A Streamlit-based analysis dashboard

### Current Prototype Workflow

```text
Side-Scan Sonar Image
          ↓
Image Preprocessing
          ↓
AI Analysis
          ↓
Potential Target / Anomaly Regions
          ↓
Confidence Estimation
          ↓
Detection Visualization
          ↓
CSV / JSON Results
          ↓
Ocean X Dashboard

git clone YOUR_GITHUB_REPOSITORY_URL
cd Ocean-X-SIH26057
