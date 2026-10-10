# Gesture-Controlled Sliding Puzzle Game

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0097A7?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array_Processing-013243?style=for-the-badge&logo=numpy&logoColor=white)

**An interactive 3x3 sliding puzzle game powered by real-time computer vision and hand landmark tracking.**

</div>

---

## 🌟 Overview

Capture your photo directly through the webcam, slice it into an interactive 3x3 puzzle grid, and solve it purely through natural hand gestures in mid-air—no mouse or keyboard required!

The project includes two gesture control mechanics:
1. **Grab & Drop Mode (`Game.py`):** Open hand (5 fingers) grabs a tile; fist (closed hand) drops and swaps it with the destination tile.
2. **Precision Pinch Mode (`PinchMove.py`):** Dynamic pinch tracking (distance between thumb and index fingertips) to grab, drag across the board, and release to drop.

---

## 🎮 Gameplay & Controls

### Capture Stage
* **Webcam Mirror Preview:** Real-time selfie framing.
* **`SPACE`:** Start a 3-second countdown and auto-capture the puzzle image.
* **`ESC`:** Abort capture or exit the application.

### In-Game Gestures

| Action | Grab & Drop Mode (`Game.py`) | Pinch Mode (`PinchMove.py`) |
| :--- | :--- | :--- |
| **Grab Tile** | Hover open hand (5 fingers) over tile | Pinch thumb tip + index fingertip |
| **Drag Tile** | Move open hand across HUD | Move pinched hand |
| **Drop & Swap** | Make a fist (0 fingers extended) | Release pinch (separate fingers) |
| **Reshuffle** | Press `R` on keyboard | Press `R` on keyboard |
| **Quit** | Press `ESC` | Press `ESC` |

---

## 🛠️ Technical Details

* **Hand Landmark Tracking:** Google MediaPipe Hands tracking 21 3D hand landmarks in real time.
* **Image Processing:** OpenCV (`cv2`) for frame capture, color conversion, tile extraction, and dynamic HUD overlay rendering.
* **Puzzle Logic:** Matrix inversion counting to guarantee all shuffled configurations are 100% mathematically solvable.
* **Animation & Visual Feedback:** Dynamic blue glowing target highlighting and tile swap animations.

---

## 🚀 Quickstart

### Prerequisites

* Python 3.8 or higher
* Working webcam

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VivekJangam126/gesture-sliding-puzzle.git
   cd gesture-sliding-puzzle
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the game:**
   * **For Grab & Drop mode:**
     ```bash
     python Game.py
     ```
   * **For Precision Pinch mode:**
     ```bash
     python PinchMove.py
     ```

---

## 📄 License

MIT License. Created by [Vivek Jangam](https://github.com/VivekJangam126).
