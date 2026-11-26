<div align="center">

<img src="assets/banner.svg" alt="DL-PR4B Banner" width="100%"/>

# 🧠 DL-PR4B — Deep Learning Practical 4B

**One project. Three senses. Infinite curiosity.**
*A multimodal deep-learning suite that can **see** images, **hear** audio and **understand** human body pose — then fuses them into a single smart application.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Nets-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-✔-00f5d4?style=for-the-badge)
![Audio AI](https://img.shields.io/badge/Audio%20AI-✔-9b5de5?style=for-the-badge)

[🚀 Quick Start](#-quick-start) • [🧩 Modules](#-the-four-modules) • [🏗 Architecture](#-architecture) • [📸 Results](#-results-gallery) • [📂 Structure](#-repository-structure)

</div>

---

## ✨ Why this project is different

| 🎯 Typical practical | 🔥 DL-PR4B |
|---|---|
| One data type | **Three modalities** — vision, sound, motion |
| Notebook output only | **Interactive Streamlit web app** |
| Separate experiments | **Combined application** that fuses all of them |
| Text results | **Visual proof** — screenshots of every result |

---

## 🧩 The Four Modules

```mermaid
mindmap
  root((DL-PR4B))
    Image
      Image classification
      CNN based
    Audio
      Audio classification
      Spectrogram / MFCC
    Pose
      Human pose estimation
      Body keypoints
    Combined
      Image + Audio + Pose
      One unified pipeline
    Streamlit
      Live web interface
      Upload and predict
```

| Module | What it does | Input | Output |
|:--:|---|---|---|
| 🖼 **Image** | Recognises what is inside a picture using a deep neural network | `.jpg` / `.png` | Class label + confidence |
| 🎧 **Audio** | Listens to a sound clip and identifies it | `.wav` / `.mp3` | Predicted sound class |
| 🕺 **Pose** | Detects the human body and tracks keypoints | Image / webcam | Skeleton overlay |
| 🔗 **Combined** | Merges all three senses into one application | Mixed | Unified result |

> 📖 Module details → [`DL_PR4B/README.md`](DL_PR4B/README.md)
> 🌐 Web app guide → [`streamlit/README.md`](streamlit/README.md)

---

## 🏗 Architecture

```mermaid
flowchart LR
    A[👤 User] --> B{Input Type}
    B -->|Picture| C[🖼 Image Model]
    B -->|Sound| D[🎧 Audio Model]
    B -->|Body / Video| E[🕺 Pose Model]
    C --> F[🔗 Combined Application]
    D --> F
    E --> F
    F --> G[🌐 Streamlit UI]
    G --> H[📊 Predictions + Visuals]
    style A fill:#9b5de5,color:#fff
    style F fill:#f15bb5,color:#fff
    style G fill:#FF4B4B,color:#fff
    style H fill:#00f5d4,color:#000
```

---

## 🚀 Quick Start

```bash
# 1️⃣ Clone
git clone https://github.com/hetttk/DL-PR4B-Deep-Learning.git
cd DL-PR4B-Deep-Learning

# 2️⃣ Virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt   # use the requirements file of the module you run

# 4️⃣ Launch the web app
cd streamlit
streamlit run app.py              # change app.py to your real file name
```

Open **http://localhost:8501** 🎉

---

## 📂 Repository Structure

```text
DL-PR4B-Deep-Learning/
│
├── 📁 DL_PR4B/            ← Core deep-learning work
│   └── README.md            (image · audio · pose · combined application)
│
├── 📁 streamlit/          ← Interactive web application
│   └── README.md
│
├── 📁 screenshots/        ← Result screenshots of both projects
│   ├── dl_pr4b/
│   └── streamlit/
│
├── 📁 assets/             ← Banner & design assets
├── 📄 .gitignore
└── 📄 README.md           ← You are here
```

---

## 📸 Results Gallery

Quick preview — full gallery in [`screenshots/README.md`](screenshots/README.md).

| 🧪 DL_PR4B | 🌐 Streamlit |
|:--:|:--:|
| ![DL_PR4B result](screenshots/dl_pr4b/combined_result.png) | ![Streamlit result](screenshots/streamlit/home.png) |

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras or PyTorch *(edit to match your code)* |
| Vision & Pose | OpenCV, MediaPipe *(edit)* |
| Audio | Librosa *(edit)* |
| Web UI | Streamlit |
| Data & Plots | NumPy, Pandas, Matplotlib |

---

## 🎓 What I Learned

- Building neural networks for **three different data types**
- Turning raw audio into model-ready features
- **Human pose estimation**
- Fusing separate models into **one product**
- Deploying ML with a clean **Streamlit** interface

---

## 👨‍💻 Author

**hetttk** — [GitHub](https://github.com/hetttk)

<div align="center">

⭐ *If this project impressed you, drop a star!* ⭐

</div>
