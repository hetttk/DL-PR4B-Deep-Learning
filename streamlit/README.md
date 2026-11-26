<div align="center">

# 🌐 Streamlit App — Deep Learning, Made Interactive

**Upload. Click. Watch the AI think.**

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-Interactive-00f5d4?style=for-the-badge)

</div>

> ⬅️ Back to [Root README](../README.md) • 🧪 Core models: [DL_PR4B](../DL_PR4B/README.md)

## 🎯 What is this?

The **Streamlit application** wraps the deep-learning models in a clean web interface, so anyone can test them **without writing a single line of code**.

## 🎬 User journey

```mermaid
journey
    title Using the Streamlit App
    section Open
      Launch app: 5: User
    section Interact
      Choose a module: 5: User
      Upload image / audio / pose input: 4: User
      Click Predict: 5: User
    section Result
      See prediction instantly: 5: App
      Review visual output: 5: User
```

## ✨ Features

| Feature | Description |
|---|---|
| 🧭 **Sidebar navigation** | Switch between modules instantly |
| 📤 **File upload** | Drag & drop images and audio |
| ⚡ **Instant predictions** | Results in seconds |
| 🎨 **Visual output** | Labels, confidence and skeleton overlays |
| 🧩 **Modular code** | Easy to extend with new models |

## 🚀 Run it locally

```bash
cd streamlit
pip install streamlit
pip install -r requirements.txt
streamlit run app.py        # use your actual main file name
```

Open **http://localhost:8501** 🎉

## 🖥 App preview

| Home | Prediction |
|:--:|:--:|
| ![Home](../screenshots/streamlit/home.png) | ![Prediction](../screenshots/streamlit/prediction.png) |

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | `pip install streamlit` |
| Model file not found | Check the model path inside the script |
| Port busy | `streamlit run app.py --server.port 8502` |
