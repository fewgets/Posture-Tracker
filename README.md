<!-- Posture Tracker · README by Usama Shahid (fewgets) -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,55:0F3D3E,100:36FFC7&height=230&section=header&text=Posture%20Tracker&fontSize=60&fontColor=FFFFFF&fontAlignY=36&desc=Real-time%20exercise%20form%20analysis%20with%20MediaPipe&descSize=17&descAlignY=58&animation=fadeIn" width="100%" alt="Posture Tracker"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=700&color=36FFC7&center=true&vCenter=true&width=720&height=60&lines=Pose+Estimation+in+Real+Time;Automatic+Rep+Counting;Instant+Form+Feedback" alt="Typing SVG"/>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

</div>

---
## 📌 Overview

**Posture Tracker** is a computer-vision app that helps you **exercise with correct form**. It uses
**MediaPipe Pose** to track your body in real time through the webcam, calculates joint angles,
**counts your reps automatically** and gives **instant feedback** like *"Go Lower!"*, *"Too Shallow!"*
or *"Perfect Form!"*

---

## 📷 Demo

| Bicep Curl Tracking | No Person Detected |
|:-:|:-:|
| <img src="images/bicep.png" width="380"/> | <img src="images/NoPerson.png" width="380"/> |

---

## ✨ Features

- 🎥 **Real-time webcam tracking** with OpenCV
- 🧍 **33-point skeleton overlay** from MediaPipe Pose
- 🏋️ **4 exercises:** bicep curls, squats, push-ups and planks
- 🔢 **Automatic rep counting** based on joint angles
- ✅ **Live form feedback** on screen
- 🚫 **Smart detection:** warns when no person is in frame

---

## 🧠 How It Works

```mermaid
flowchart LR
    A[🎥 Webcam frame] --> B[🧍 MediaPipe Pose<br/>landmarks]
    B --> C[📐 Angle at the joint<br/>e.g. shoulder-elbow-wrist]
    C --> D{Angle in<br/>correct range?}
    D -->|Yes| E[🔢 Count rep<br/>✅ Perfect form]
    D -->|No| F[⚠️ Feedback<br/>Go lower / Too shallow]
```

The angle at a joint is calculated from three landmarks. A **rep** is counted when the angle moves
through the full range of the exercise, for example from arm extended (~160°) to fully curled (~30°).

---

## 🚀 Run Locally

```bash
git clone https://github.com/fewgets/Posture-Tracker.git
cd Posture-Tracker

pip install -r requirement.txt      # opencv-python, mediapipe, numpy
python PoseTracker.py
```

- Press **`q`** to quit the exercise window.
- Using a second camera? Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `PoseTracker.py`.

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10 |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe Pose |
| Maths | NumPy |

> 💡 This tracker later became the core of **[FitFusion](https://github.com/fewgets/FitFusion)**, a full AI fitness app with meal planning and progress tracking.

---

## 🤝 Let's Work Together

<div align="center">

**Have an idea, a freelance project, or want to collaborate on something in AI?**
I'm always open to collaborations, freelance work and full-time opportunities.

<a href="mailto:shaikhusama541@gmail.com?subject=Collaboration%20-%20Posture%20Tracker">
  <img src="https://img.shields.io/badge/Email-shaikhusama541%40gmail.com-36FFC7?style=for-the-badge&logo=gmail&logoColor=white&labelColor=0D1117" alt="Email"/>
</a>
<a href="https://www.linkedin.com/in/-usamashahid/">
  <img src="https://img.shields.io/badge/LinkedIn-Usama%20Shahid-36FFC7?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0D1117" alt="LinkedIn"/>
</a>
<a href="https://github.com/fewgets">
  <img src="https://img.shields.io/badge/GitHub-fewgets-36FFC7?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117" alt="GitHub"/>
</a>

<br/><br/>

⭐ **If you like this project, please give it a star. It really helps!** ⭐

<br/>

<h3>"Perfect form beats heavy weight."</h3>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:36FFC7,55:0F3D3E,100:0D1117&height=150&section=footer&animation=twinkling" width="100%" alt="footer"/>

</div>
