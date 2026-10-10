"""
Gesture-Controlled Sliding Puzzle Game  (v2)
=============================================
NEW CONTROLS:
  • Open hand (5 fingers) over a tile  → GRAB & DRAG that tile
  • Fist (0 fingers / closed hand)     → DROP tile; swap with tile beneath
  • Destination tile glows BLUE until the swap animation settles

CAPTURE SCREEN:
  • Live webcam preview shown first with mirror-selfie view
  • Press SPACE to start a 3-second countdown, then auto-capture
  • Press ESC on capture screen to abort

IN-GAME:
  • Press R to reshuffle
  • Press ESC to quit

Frame size: 720×720 puzzle + 260 px HUD panel = 980×720 total window

Dependencies:
  pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PUZZLE_SIZE = 720           # square puzzle area
GRID_N      = 4
TILE_SIZE   = PUZZLE_SIZE // GRID_N   # 180 px for 4x4 at 720px

HUD_W       = 260           # side panel width
WINDOW_W    = PUZZLE_SIZE + HUD_W
WINDOW_H    = PUZZLE_SIZE

LERP_SPEED  = 0.22          # tile animation smoothing (0 = frozen, 1 = instant)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_GRID       = (0,  230, 230)    # cyan grid lines
COL_BORDER     = (0,  180, 255)    # puzzle outer border
COL_DRAG_H0    = (0,  255,  80)    # hand-0 drag ring  (green)
COL_DRAG_H1    = (40, 100, 255)    # hand-1 drag ring  (orange-red)
COL_DEST_BLUE  = (220,  60,  30)   # blue flash on swapped tile (BGR → appears blue)
COL_TEXT       = (230, 230, 230)
COL_TITLE      = (0,  220, 255)
COL_SOLVED     = (0,  255, 160)
COL_FPS        = (120, 255, 120)
COL_HUD_BG     = (18,  18,  28)
COL_HINT       = (160, 200, 255)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONT2 = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def count_fingers(hand_lm, label: str) -> int:
    """
    Return number of extended fingers (0-5).
    Tip IDs: [4,8,12,16,20]; PIP IDs: [3,6,10,14,18].
    Thumb uses horizontal x-comparison (accounts for webcam mirror).
    Returns 5 for fully open hand, 0 for closed fist.
    """
    lm   = hand_lm.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up   = 0

    # Thumb (horizontal)
    if label == "Right":
        if lm[tips[0]].x < lm[pips[0]].x:
            up += 1
    else:
        if lm[tips[0]].x > lm[pips[0]].x:
            up += 1

    # Four fingers (vertical: tip above pip = extended)
    for i in range(1, 5):
        if lm[tips[i]].y < lm[pips[i]].y:
            up += 1

    return up


def palm_center_px(hand_lm, w: int, h: int):
    """
    Stable palm centre = average of wrist + 4 MCP joints (landmarks 0,5,9,13,17).
    Mapped to pixel coords (w, h).
    """
    ids = [0, 5, 9, 13, 17]
    xs  = [hand_lm.landmark[i].x for i in ids]
    ys  = [hand_lm.landmark[i].y for i in ids]
    return int(sum(xs) / len(xs) * w), int(sum(ys) / len(ys) * h)


def shadow_text(img, text, pos, font, scale, colour, thickness):
    """Draw text with a 1-pixel black drop shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), font, scale,
                colour,    thickness,   cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  TILE
# ─────────────────────────────────────────────────────────────────────────────
class Tile:
    """
    One puzzle piece.
    Tracks:
      • image crop (numpy array)
      • logical grid position  (grid_row, grid_col)
      • animated render position (render_x, render_y) that lerps toward target
      • blue_flash countdown — shows a blue border while > 0
    """

    def __init__(self, tile_id: int, image: np.ndarray, row: int, col: int):
        self.tile_id  = tile_id     # original order index (for solved check)
        self.image    = image       # TILE_SIZE × TILE_SIZE BGR
        self.grid_row = row
        self.grid_col = col

        px = float(col * TILE_SIZE)
        py = float(row * TILE_SIZE)
        self.render_x = px;  self.render_y = py
        self.target_x = px;  self.target_y = py
        self.blue_flash = 0.0       # seconds of blue highlight remaining

    # ── helpers ───────────────────────────────────────────────────────────────
    def grid_origin(self):
        return self.grid_col * TILE_SIZE, self.grid_row * TILE_SIZE

    def set_target_grid(self):
        gx, gy = self.grid_origin()
        self.target_x = float(gx);  self.target_y = float(gy)

    def snap_to_grid(self):
        gx, gy = self.grid_origin()
        self.render_x = float(gx);  self.render_y = float(gy)
        self.target_x = self.render_x;  self.target_y = self.render_y

    def set_target_palm(self, px: int, py: int):
        """Centre tile on the palm position."""
        self.target_x = float(px - TILE_SIZE // 2)
        self.target_y = float(py - TILE_SIZE // 2)

    # ── per-frame update ──────────────────────────────────────────────────────
    def update(self, dt: float):
        """Advance lerp animation and tick down the blue flash timer."""
        self.render_x += (self.target_x - self.render_x) * LERP_SPEED
        self.render_y += (self.target_y - self.render_y) * LERP_SPEED
        if self.blue_flash > 0:
            self.blue_flash = max(0.0, self.blue_flash - dt)


# ─────────────────────────────────────────────────────────────────────────────
#  PUZZLE
# ─────────────────────────────────────────────────────────────────────────────
class Puzzle:
    """
    Manages the GRID_N × GRID_N grid of Tiles.
    Responsibilities: slicing the source image, shuffling, swapping, solved-check,
    per-frame update, and rendering.
    """

    def __init__(self, source_img: np.ndarray):
        # Resize captured frame to the exact puzzle canvas size
        img = cv2.resize(source_img, (PUZZLE_SIZE, PUZZLE_SIZE))

        # Slice into GRID_N × GRID_N tiles
        self.tiles: list[Tile] = []
        tid = 0
        for r in range(GRID_N):
            for c in range(GRID_N):
                crop = img[r * TILE_SIZE:(r+1) * TILE_SIZE,
                           c * TILE_SIZE:(c+1) * TILE_SIZE].copy()
                self.tiles.append(Tile(tid, crop, r, c))
                tid += 1

        # Lookup map: (row, col) → Tile
        self.grid: dict[tuple, Tile] = {(t.grid_row, t.grid_col): t
                                         for t in self.tiles}

        self.move_count = 0
        self.start_time = time.time()
        self.solved     = False
        self.shuffle()

    # ── shuffle ───────────────────────────────────────────────────────────────
    def shuffle(self):
        """Randomly assign each tile to a unique grid cell."""
        positions = [(r, c) for r in range(GRID_N) for c in range(GRID_N)]
        random.shuffle(positions)
        for tile, (r, c) in zip(self.tiles, positions):
            tile.grid_row = r;  tile.grid_col = c
            tile.snap_to_grid()
        self.grid = {(t.grid_row, t.grid_col): t for t in self.tiles}

    # ── query ─────────────────────────────────────────────────────────────────
    def tile_at_pixel(self, px: int, py: int) -> "Tile | None":
        """Return the tile whose logical grid cell contains pixel (px, py)."""
        col = max(0, min(GRID_N - 1, px // TILE_SIZE))
        row = max(0, min(GRID_N - 1, py // TILE_SIZE))
        return self.grid.get((row, col))

    # ── swap ──────────────────────────────────────────────────────────────────
    def swap(self, a: Tile, b: Tile):
        """
        Swap tiles a and b in the grid.
        Tile b (the destination) gets a blue-flash highlight for 0.9 seconds.
        """
        if a is b:
            return
        del self.grid[(a.grid_row, a.grid_col)]
        del self.grid[(b.grid_row, b.grid_col)]

        a.grid_row, b.grid_row = b.grid_row, a.grid_row
        a.grid_col, b.grid_col = b.grid_col, a.grid_col

        self.grid[(a.grid_row, a.grid_col)] = a
        self.grid[(b.grid_row, b.grid_col)] = b

        a.set_target_grid()
        b.set_target_grid()
        b.blue_flash = 0.9      # ← blue highlight on the tile that got swapped in
        self.move_count += 1

    # ── solved check ──────────────────────────────────────────────────────────
    def check_solved(self) -> bool:
        """Returns True when every tile is back in its original (tile_id) cell."""
        for t in self.tiles:
            if t.grid_row != t.tile_id // GRID_N or t.grid_col != t.tile_id % GRID_N:
                return False
        return True

    # ── update ────────────────────────────────────────────────────────────────
    def update(self, dragging: set, dt: float):
        """
        Move non-dragged tiles toward their grid targets.
        Dragged tiles are moved by the HandState FSM instead.
        """
        for tile in self.tiles:
            if tile not in dragging:
                tile.set_target_grid()
            tile.update(dt)

    # ── render ────────────────────────────────────────────────────────────────
    def render(self, canvas: np.ndarray, drag_highlights: dict):
        """
        Blit all tiles onto canvas, then draw borders and grid lines.

        draw order:
          1. Non-dragged tiles  (drawn first so dragged tile floats on top)
          2. Dragged tiles

        drag_highlights: {Tile: BGR colour} for currently held tiles.
        """
        dragged_set = set(drag_highlights.keys())
        draw_order  = ([t for t in self.tiles if t not in dragged_set] +
                       [t for t in self.tiles if t in dragged_set])

        for tile in draw_order:
            # Clamp render position to stay within canvas
            x = max(0, min(PUZZLE_SIZE - TILE_SIZE, int(round(tile.render_x))))
            y = max(0, min(PUZZLE_SIZE - TILE_SIZE, int(round(tile.render_y))))

            # Blit tile image
            canvas[y:y + TILE_SIZE, x:x + TILE_SIZE] = tile.image

            # Blue flash border (destination tile after a swap)
            if tile.blue_flash > 0 and tile not in dragged_set:
                # Fade: full opacity for first 0.4 s, then fade out
                alpha  = min(1.0, tile.blue_flash / 0.4)
                border = max(1, int(7 * alpha))
                cv2.rectangle(canvas,
                              (x + 2, y + 2),
                              (x + TILE_SIZE - 2, y + TILE_SIZE - 2),
                              COL_DEST_BLUE, border)

            # Drag highlight (coloured border while being held)
            if tile in drag_highlights:
                cv2.rectangle(canvas,
                              (x + 3, y + 3),
                              (x + TILE_SIZE - 3, y + TILE_SIZE - 3),
                              drag_highlights[tile], 5)

        # Yellow-cyan grid lines
        for i in range(1, GRID_N):
            p = i * TILE_SIZE
            cv2.line(canvas, (p, 0),          (p, PUZZLE_SIZE), COL_GRID, 2)
            cv2.line(canvas, (0, p),          (PUZZLE_SIZE, p), COL_GRID, 2)

        # Outer border
        cv2.rectangle(canvas, (0, 0),
                      (PUZZLE_SIZE - 1, PUZZLE_SIZE - 1), COL_BORDER, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  HAND STATE
# ─────────────────────────────────────────────────────────────────────────────
class HandState:
    """
    Finite-state machine for one detected hand.

    States:
      IDLE   : waiting for an open hand (5 fingers)
      DRAG   : holding a tile, following the palm
      DROP   : fist detected → swap and return to IDLE

    Transition table:
      IDLE  + fingers==5  → grab tile under palm  → DRAG
      DRAG  + fingers==5  → continue dragging
      DRAG  + fingers==0  → drop, swap             → IDLE
      DRAG  + 1-4 fingers → keep dragging (avoids accidental drop mid-gesture)
      any   + hand gone   → release tile           → IDLE
    """

    def __init__(self, drag_colour):
        self.drag_colour          = drag_colour
        self.holding: Tile | None = None

    def process(self, fingers: int, palm_x: int, palm_y: int, puzzle: Puzzle):
        if fingers == 5:
            if self.holding is None:
                # Attempt grab
                tile = puzzle.tile_at_pixel(palm_x, palm_y)
                if tile:
                    self.holding = tile
            else:
                # Continue drag
                self.holding.set_target_palm(palm_x, palm_y)

        elif fingers == 0:
            # Fist → drop
            if self.holding is not None:
                cx   = int(round(self.holding.render_x)) + TILE_SIZE // 2
                cy   = int(round(self.holding.render_y)) + TILE_SIZE // 2
                dest = puzzle.tile_at_pixel(cx, cy)
                if dest and dest is not self.holding:
                    puzzle.swap(self.holding, dest)
                else:
                    self.holding.set_target_grid()  # no valid swap → return home
                self.holding = None

        else:
            # 1–4 fingers: keep dragging if we were already holding something
            if self.holding is not None:
                self.holding.set_target_palm(palm_x, palm_y)

    def release(self):
        """Called when the hand disappears from frame — return tile home."""
        if self.holding:
            self.holding.set_target_grid()
            self.holding = None


# ─────────────────────────────────────────────────────────────────────────────
#  CAPTURE SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def capture_screen(cap) -> "np.ndarray | None":
    """
    Show a live mirror-selfie preview.
    SPACE  → starts a 3-second countdown, then captures and returns the frame.
    ESC    → returns None (abort).
    """
    print("[Capture] Press SPACE to start countdown, ESC to cancel.")
    countdown_start = None   # None = idle;  float = timestamp when SPACE pressed

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame   = cv2.flip(frame, 1)         # mirror for natural selfie view
        display = frame.copy()
        h, w    = display.shape[:2]

        # Darkened instruction banner at the bottom
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        if countdown_start is None:
            # ── Waiting for SPACE ────────────────────────────────────────────
            shadow_text(display, "Frame yourself in the preview",
                        (24, h - 90), FONT, 0.7, COL_TITLE, 2)
            shadow_text(display, "Press  SPACE  to capture puzzle image",
                        (24, h - 55), FONT, 0.72, COL_TEXT,  2)
            shadow_text(display, "Press  ESC  to quit",
                        (24, h - 20), FONT, 0.55, (120, 120, 200), 1)
        else:
            # ── Counting down ────────────────────────────────────────────────
            elapsed   = time.time() - countdown_start
            remaining = 3.0 - elapsed
            if remaining <= 0:
                cv2.destroyWindow("Gesture Puzzle — Capture")
                return frame            # ← return captured frame
            num = int(remaining) + 1   # display 3, 2, 1 …
            shadow_text(display, f"Capturing in  {num}…",
                        (24, h - 60), FONT2, 1.3, COL_SOLVED, 3)

        cv2.imshow("Gesture Puzzle — Capture", display)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:                      # ESC
            cv2.destroyWindow("Gesture Puzzle — Capture")
            return None
        elif key == ord(' ') and countdown_start is None:
            countdown_start = time.time()  # start countdown


# ─────────────────────────────────────────────────────────────────────────────
#  HUD PANEL
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(panel: np.ndarray,
             puzzle: Puzzle,
             fps: float,
             hand_states: list,
             detected_hands: list):
    """Render all informational text onto the right-hand side panel."""
    panel[:] = COL_HUD_BG    # clear with dark background
    W = panel.shape[1]
    y = 38

    # ── Title ─────────────────────────────────────────────────────────────────
    shadow_text(panel, "GESTURE", (14, y),      FONT2, 0.85, COL_TITLE, 2)
    shadow_text(panel, "PUZZLE",  (14, y + 34), FONT2, 0.85, COL_TITLE, 2)
    cv2.line(panel, (10, y + 52), (W - 10, y + 52), (60, 60, 90), 1)
    y += 74

    # ── Stats ─────────────────────────────────────────────────────────────────
    elapsed = int(time.time() - puzzle.start_time)
    mm, ss  = divmod(elapsed, 60)
    shadow_text(panel, f"Moves : {puzzle.move_count}", (14, y),      FONT, 0.58, COL_TEXT, 1)
    shadow_text(panel, f"Time  : {mm:02d}:{ss:02d}",  (14, y + 28), FONT, 0.58, COL_TEXT, 1)
    shadow_text(panel, f"FPS   : {fps:.1f}",           (14, y + 56), FONT, 0.58, COL_FPS,  1)
    y += 82
    cv2.line(panel, (10, y), (W - 10, y), (60, 60, 90), 1)
    y += 18

    # ── Controls ──────────────────────────────────────────────────────────────
    shadow_text(panel, "CONTROLS", (14, y), FONT, 0.55, COL_TITLE, 1)
    y += 24
    controls = [
        ("Open hand (5)", "Grab & drag tile"),
        ("Fist (0 fingers)", "Drop & swap"),
        ("[R] key", "Reshuffle"),
        ("[ESC]", "Quit"),
    ]
    for key_t, val_t in controls:
        shadow_text(panel, key_t, (14, y),      FONT, 0.43, COL_HINT, 1)
        shadow_text(panel, val_t, (14, y + 17), FONT, 0.42, COL_TEXT, 1)
        y += 40
    cv2.line(panel, (10, y), (W - 10, y), (60, 60, 90), 1)
    y += 18

    # ── Hand status ───────────────────────────────────────────────────────────
    shadow_text(panel, "HAND STATUS", (14, y), FONT, 0.52, COL_TITLE, 1)
    y += 24
    for i, (hs, hinfo) in enumerate(zip(hand_states, detected_hands)):
        col = hs.drag_colour
        if hinfo:
            fingers, _, _, label = hinfo
            if hs.holding:
                status = "HOLDING tile"
            elif fingers == 5:
                status = "Open (grab)"
            elif fingers == 0:
                status = "Fist (drop)"
            else:
                status = f"{fingers} finger{'s' if fingers != 1 else ''}"
            shadow_text(panel, f"Hand {i+1} ({label[0]}): {status}",
                        (14, y), FONT, 0.42, col, 1)
        else:
            shadow_text(panel, f"Hand {i+1}: not detected",
                        (14, y), FONT, 0.42, (70, 70, 70), 1)
        y += 24
    cv2.line(panel, (10, y), (W - 10, y), (60, 60, 90), 1)
    y += 18

    # ── Solved banner ─────────────────────────────────────────────────────────
    if puzzle.solved:
        cv2.rectangle(panel, (6,  y),      (W - 6, y + 66), (0, 70, 0), -1)
        cv2.rectangle(panel, (6,  y),      (W - 6, y + 66), COL_SOLVED, 2)
        shadow_text(panel, "SOLVED!", (18, y + 26), FONT2, 0.95, COL_SOLVED, 2)
        shadow_text(panel, f"{mm:02d}:{ss:02d} / {puzzle.move_count} moves",
                    (14, y + 54), FONT, 0.45, COL_TEXT, 1)
        y += 80

    # ── Legend at the bottom ──────────────────────────────────────────────────
    by = panel.shape[0] - 30
    cv2.circle(panel, (18, by - 4), 7, COL_DEST_BLUE, -1)
    shadow_text(panel, "= swapped tile", (30, by), FONT, 0.43, COL_TEXT, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── MediaPipe Hands setup ─────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode         = False,
        max_num_hands             = 2,
        min_detection_confidence  = 0.72,
        min_tracking_confidence   = 0.65,
    )

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ── Capture screen ────────────────────────────────────────────────────────
    puzzle_img = capture_screen(cap)
    if puzzle_img is None:
        print("Capture cancelled. Goodbye!")
        cap.release()
        cv2.destroyAllWindows()
        return

    # ── Build puzzle ──────────────────────────────────────────────────────────
    puzzle = Puzzle(puzzle_img)

    # ── Two independent hand states ──────────────────────────────────────────
    hand_states = [HandState(COL_DRAG_H0), HandState(COL_DRAG_H1)]

    # ── FPS / timing ─────────────────────────────────────────────────────────
    fps_display = 0.0
    fps_counter = 0
    fps_timer   = time.time()
    prev_time   = time.time()

    print("Game started!  Open hand = grab tile.  Fist = drop tile.  ESC = quit.")

    # ── Main game loop ────────────────────────────────────────────────────────
    while True:
        ret, raw = cap.read()
        if not ret:
            break

        # Delta time for smooth animation regardless of frame rate
        now       = time.time()
        dt        = now - prev_time
        prev_time = now

        # Mirror the webcam frame
        raw = cv2.flip(raw, 1)

        # ── Hand detection ────────────────────────────────────────────────────
        rgb     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        # Build list of (fingers, palm_x, palm_y, label) — one entry per hand
        detected_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hlm, hcls in zip(results.multi_hand_landmarks,
                                  results.multi_handedness):
                label   = hcls.classification[0].label
                fingers = count_fingers(hlm, label)
                px, py  = palm_center_px(hlm, PUZZLE_SIZE, PUZZLE_SIZE)
                detected_hands.append((fingers, px, py, label))

        # Always keep exactly 2 slots (pad with None)
        while len(detected_hands) < 2:
            detected_hands.append(None)
        detected_hands = detected_hands[:2]

        # ── Update hand FSMs ──────────────────────────────────────────────────
        for i, hinfo in enumerate(detected_hands):
            if hinfo:
                fingers, px, py, _ = hinfo
                hand_states[i].process(fingers, px, py, puzzle)
            else:
                hand_states[i].release()   # hand lost → return held tile

        # ── Puzzle physics / animation ────────────────────────────────────────
        dragging = {hs.holding for hs in hand_states if hs.holding}
        puzzle.update(dragging, dt)

        if not puzzle.solved:
            puzzle.solved = puzzle.check_solved()

        # ── Compose the full window ───────────────────────────────────────────
        # Left  PUZZLE_SIZE×PUZZLE_SIZE: puzzle area
        # Right HUD_W×PUZZLE_SIZE      : information panel
        window = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        puzzle_canvas = window[:, :PUZZLE_SIZE]   # view into left section
        hud_panel     = window[:, PUZZLE_SIZE:]   # view into right section

        # Draw puzzle
        drag_highlights = {hs.holding: hs.drag_colour
                           for hs in hand_states if hs.holding}
        puzzle.render(puzzle_canvas, drag_highlights)

        # Palm cursor overlays
        for i, hinfo in enumerate(detected_hands):
            if hinfo:
                fingers, px, py, _ = hinfo
                col = hand_states[i].drag_colour
                # Outer ring
                cv2.circle(puzzle_canvas, (px, py), 20, col, 3)
                # Inner filled dot
                cv2.circle(puzzle_canvas, (px, py),  7, col, -1)
                # Finger count
                shadow_text(puzzle_canvas, str(fingers),
                            (px + 24, py - 8), FONT, 0.8, col, 2)
                # Gesture label
                if fingers == 5:
                    lbl = "OPEN"
                elif fingers == 0:
                    lbl = "FIST"
                else:
                    lbl = ""
                if lbl:
                    shadow_text(puzzle_canvas, lbl,
                                (px + 24, py + 18), FONT, 0.6, col, 1)

        # Solved overlay
        if puzzle.solved:
            ovl = puzzle_canvas.copy()
            cv2.rectangle(ovl, (0, 0), (PUZZLE_SIZE, PUZZLE_SIZE), (0,0,0), -1)
            cv2.addWeighted(ovl, 0.35, puzzle_canvas, 0.65, 0, puzzle_canvas)
            banner = "PUZZLE  SOLVED!"
            (tw, th), _ = cv2.getTextSize(banner, FONT2, 1.7, 3)
            bx = (PUZZLE_SIZE - tw) // 2
            by = PUZZLE_SIZE // 2 + th // 2
            cv2.putText(puzzle_canvas, banner, (bx+3, by+3),
                        FONT2, 1.7, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(puzzle_canvas, banner, (bx,   by),
                        FONT2, 1.7, COL_SOLVED, 3, cv2.LINE_AA)

        # Draw HUD
        draw_hud(hud_panel, puzzle, fps_display, hand_states, detected_hands)

        # Separator line between puzzle and HUD
        cv2.line(window, (PUZZLE_SIZE, 0), (PUZZLE_SIZE, WINDOW_H), (70, 70, 100), 2)

        cv2.imshow("Gesture Puzzle v2  —  ESC to quit", window)

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()

        # ── Keyboard shortcuts ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                   # ESC → quit
            break
        elif key == ord('r'):           # R → reshuffle
            puzzle.shuffle()
            puzzle.solved      = False
            puzzle.move_count  = 0
            puzzle.start_time  = time.time()
            for hs in hand_states:
                hs.release()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()