"""
╔══════════════════════════════════════════════════════════════════╗
║        GESTURE-CONTROLLED 3×3 PINCH PUZZLE GAME                 ║
║  OpenCV  •  MediaPipe Hands  •  NumPy                           ║
╠══════════════════════════════════════════════════════════════════╣
║  CONTROLS                                                        ║
║    SPACE  ── capture image & start puzzle                        ║
║    ESC    ── quit at any time                                    ║
║    R      ── reshuffle during game                               ║
║                                                                  ║
║  GESTURES                                                        ║
║    Pinch (thumb tip + index tip close) ── grab tile              ║
║    Move hand while pinched             ── drag tile              ║
║    Release pinch                       ── drop & swap            ║
╚══════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PUZZLE_SIZE    = 450          # puzzle canvas is always 450 × 450 px
GRID_N         = 3            # 3 × 3 grid
TILE_SIZE      = PUZZLE_SIZE // GRID_N   # 150 px per tile

PINCH_THRESH   = 40           # pixel distance to trigger pinch
LERP_ALPHA     = 0.20         # smoothing factor for tile drag (0 = frozen, 1 = instant)

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
C_GRID         = (0, 200, 255)    # yellow / gold grid lines
C_BORDER       = (255, 200,  50)  # outer puzzle border
C_SELECT       = (0, 255, 80)     # selected-tile highlight
C_PINCH_ACTIVE = (0, 255, 80)     # cursor ring when pinching
C_PINCH_IDLE   = (200, 200, 200)  # cursor ring when open
C_TEXT         = (255, 255, 255)
C_SHADOW       = (0, 0, 0)
C_SOLVED       = (0, 255, 160)
C_HUD_BG       = (20, 20, 30)
C_FPS          = (120, 255, 100)
C_CAPTION      = (0, 220, 255)

FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def shadow_text(img, text, pos, font, scale, colour, thickness, shadow=C_SHADOW):
    """Draw text with a 1-pixel drop shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, shadow,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y    ), font, scale, colour,
                thickness,     cv2.LINE_AA)


def pinch_distance(hand_lm, frame_w: int, frame_h: int) -> tuple:
    """
    Compute the Euclidean pixel distance between the THUMB TIP (landmark 4)
    and INDEX FINGER TIP (landmark 8).

    Returns:
        distance  – float, pixels
        mid_x     – int,   midpoint x pixel
        mid_y     – int,   midpoint y pixel
    """
    lm  = hand_lm.landmark
    # Map normalised coordinates → pixel coordinates
    tx  = lm[4].x * frame_w;   ty = lm[4].y * frame_h
    ix  = lm[8].x * frame_w;   iy = lm[8].y * frame_h
    dist = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)
    mid_x = int((tx + ix) / 2)
    mid_y = int((ty + iy) / 2)
    return dist, mid_x, mid_y


# ═══════════════════════════════════════════════════════════════════════════════
#  TILE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Tile:
    """
    Represents one puzzle piece.

    Attributes:
        tile_id   – original position index (used for solved check)
        image     – TILE_SIZE × TILE_SIZE BGR numpy crop
        grid_row  – current logical row in the grid
        grid_col  – current logical column in the grid
        render_x  – animated top-left x (lerped toward target)
        render_y  – animated top-left y (lerped toward target)
        target_x  – where the tile should move toward
        target_y  – (set by hand drag or grid snap)
    """

    def __init__(self, tile_id: int, image: np.ndarray, row: int, col: int):
        self.tile_id  = tile_id
        self.image    = image
        self.grid_row = row
        self.grid_col = col

        # Initialise render position to exact grid position
        px, py = self._grid_px()
        self.render_x = float(px)
        self.render_y = float(py)
        self.target_x = float(px)
        self.target_y = float(py)

    # ── Grid helpers ────────────────────────────────────────────────────────────
    def _grid_px(self):
        """Return the pixel top-left corner of this tile's logical grid cell."""
        return self.grid_col * TILE_SIZE, self.grid_row * TILE_SIZE

    def set_target_grid(self):
        """Point the target toward the logical grid cell (snap-back)."""
        gx, gy = self._grid_px()
        self.target_x = float(gx)
        self.target_y = float(gy)

    def snap_to_grid(self):
        """Instantly place the tile at its grid cell (no animation)."""
        gx, gy = self._grid_px()
        self.render_x = float(gx);  self.render_y = float(gy)
        self.target_x = float(gx);  self.target_y = float(gy)

    def set_target_cursor(self, cx: int, cy: int):
        """Centre the tile on the pinch midpoint cursor."""
        self.target_x = float(cx - TILE_SIZE // 2)
        self.target_y = float(cy - TILE_SIZE // 2)

    # ── Per-frame animation ─────────────────────────────────────────────────────
    def update_lerp(self):
        """Advance the render position toward the target using linear interpolation."""
        self.render_x += (self.target_x - self.render_x) * LERP_ALPHA
        self.render_y += (self.target_y - self.render_y) * LERP_ALPHA


# ═══════════════════════════════════════════════════════════════════════════════
#  PUZZLE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Puzzle:
    """
    Manages the 3×3 sliding-puzzle state.

    Responsibilities:
        • Slice source image into tiles
        • Shuffle tiles randomly
        • Handle tile queries, swaps, and solved detection
        • Render tiles + grid onto a canvas
    """

    def __init__(self, source_img: np.ndarray):
        # Resize source to exact puzzle canvas size
        img = cv2.resize(source_img, (PUZZLE_SIZE, PUZZLE_SIZE))

        # ── Slice into GRID_N × GRID_N tiles ────────────────────────────────
        self.tiles: list[Tile] = []
        tid = 0
        for r in range(GRID_N):
            for c in range(GRID_N):
                crop = img[r * TILE_SIZE:(r + 1) * TILE_SIZE,
                           c * TILE_SIZE:(c + 1) * TILE_SIZE].copy()
                self.tiles.append(Tile(tid, crop, r, c))
                tid += 1

        # Keep original order for solved check
        self.original_order = [t.tile_id for t in self.tiles]  # [0,1,2,…,8]

        # Grid lookup map: (row, col) → Tile
        self.grid: dict[tuple, Tile] = {(t.grid_row, t.grid_col): t
                                         for t in self.tiles}

        # Game stats
        self.move_count = 0
        self.start_time = time.time()
        self.solved      = False

        # Shuffle after creation
        self.shuffle()

    # ── Shuffle ──────────────────────────────────────────────────────────────
    def shuffle(self):
        """Randomly assign each tile to a unique grid cell and snap into place."""
        positions = [(r, c) for r in range(GRID_N) for c in range(GRID_N)]
        random.shuffle(positions)
        for tile, (r, c) in zip(self.tiles, positions):
            tile.grid_row = r
            tile.grid_col = c
            tile.snap_to_grid()
        self.grid = {(t.grid_row, t.grid_col): t for t in self.tiles}

    # ── Grid query ────────────────────────────────────────────────────────────
    def tile_at_pixel(self, px: int, py: int) -> "Tile | None":
        """
        Return the Tile whose logical grid cell contains pixel (px, py).
        Clamps out-of-bounds pixels to the nearest cell.
        """
        col = max(0, min(GRID_N - 1, px // TILE_SIZE))
        row = max(0, min(GRID_N - 1, py // TILE_SIZE))
        return self.grid.get((row, col))

    # ── Swap ─────────────────────────────────────────────────────────────────
    def swap(self, tile_a: Tile, tile_b: Tile):
        """
        Swap the logical grid positions of two tiles.
        Both tiles will then smoothly lerp toward their new cells.
        """
        if tile_a is tile_b:
            return

        # Remove from grid
        del self.grid[(tile_a.grid_row, tile_a.grid_col)]
        del self.grid[(tile_b.grid_row, tile_b.grid_col)]

        # Exchange positions
        tile_a.grid_row, tile_b.grid_row = tile_b.grid_row, tile_a.grid_row
        tile_a.grid_col, tile_b.grid_col = tile_b.grid_col, tile_a.grid_col

        # Re-insert into grid
        self.grid[(tile_a.grid_row, tile_a.grid_col)] = tile_a
        self.grid[(tile_b.grid_row, tile_b.grid_col)] = tile_b

        # Animate toward new positions
        tile_a.set_target_grid()
        tile_b.set_target_grid()

        self.move_count += 1

    # ── Solved check ──────────────────────────────────────────────────────────
    def check_solved(self) -> bool:
        """
        Returns True when every tile occupies its original (tile_id) cell.
        tile_id 0 → (row 0, col 0),  tile_id 5 → (row 1, col 2), etc.
        """
        for tile in self.tiles:
            expected_row = tile.tile_id // GRID_N
            expected_col = tile.tile_id  % GRID_N
            if tile.grid_row != expected_row or tile.grid_col != expected_col:
                return False
        return True

    # ── Update (called every frame) ───────────────────────────────────────────
    def update(self, dragging_tile: "Tile | None"):
        """
        Advance lerp for all tiles.
        Dragging tile is driven by the hand cursor, so we skip its grid target.
        """
        for tile in self.tiles:
            if tile is not dragging_tile:
                tile.set_target_grid()
            tile.update_lerp()

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, canvas: np.ndarray, selected_tile: "Tile | None"):
        """
        Draw all tiles, yellow grid lines, and outer border onto canvas.
        Selected tile is drawn LAST so it floats above the others.

        Args:
            canvas        – 450 × 450 black numpy array to draw onto
            selected_tile – Tile currently being dragged (drawn on top)
        """
        # ── Draw tiles ───────────────────────────────────────────────────────
        # Non-selected first, selected last (z-order)
        draw_order = ([t for t in self.tiles if t is not selected_tile] +
                      ([selected_tile] if selected_tile else []))

        for tile in draw_order:
            # Clamp render position to canvas boundaries
            x = int(round(tile.render_x))
            y = int(round(tile.render_y))
            x = max(0, min(PUZZLE_SIZE - TILE_SIZE, x))
            y = max(0, min(PUZZLE_SIZE - TILE_SIZE, y))

            # Blit tile image
            canvas[y:y + TILE_SIZE, x:x + TILE_SIZE] = tile.image

            # Highlight border for selected tile
            if tile is selected_tile:
                cv2.rectangle(canvas,
                              (x + 3,              y + 3),
                              (x + TILE_SIZE - 3,  y + TILE_SIZE - 3),
                              C_SELECT, 4)

        # ── Yellow grid lines ─────────────────────────────────────────────────
        for i in range(1, GRID_N):
            pos = i * TILE_SIZE
            cv2.line(canvas, (pos, 0),         (pos, PUZZLE_SIZE), C_GRID, 2)
            cv2.line(canvas, (0,   pos),        (PUZZLE_SIZE, pos), C_GRID, 2)

        # ── Outer border ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0),
                      (PUZZLE_SIZE - 1, PUZZLE_SIZE - 1), C_BORDER, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  HAND / PINCH STATE
# ═══════════════════════════════════════════════════════════════════════════════

class PinchHand:
    """
    Tracks the pinch state of one detected hand.

    FSM:
        IDLE   + pinch detected over tile  → grab tile → DRAGGING
        DRAGGING + pinch held              → move tile
        DRAGGING + pinch released          → drop tile → swap → IDLE
        any    + hand lost                 → release tile → IDLE
    """

    def __init__(self):
        self.holding: Tile | None = None   # tile currently being dragged

    def process(self, distance: float, mid_x: int, mid_y: int,
                puzzle: Puzzle, global_drag: "Tile | None") -> "Tile | None":
        """
        Process one hand's pinch state for this frame.

        Args:
            distance    – current thumb-index distance in pixels
            mid_x/mid_y – midpoint between thumb and index tips
            puzzle      – the puzzle object
            global_drag – the tile currently dragged by ANY hand (None if none)

        Returns:
            The tile this hand is currently dragging (or None).
        """
        is_pinching = distance < PINCH_THRESH

        if is_pinching:
            if self.holding is None:
                # ── Grab: only if no other hand is already dragging ───────────
                if global_drag is None:
                    tile = puzzle.tile_at_pixel(mid_x, mid_y)
                    if tile:
                        self.holding = tile
            else:
                # ── Drag: keep moving the tile toward the cursor ──────────────
                self.holding.set_target_cursor(mid_x, mid_y)
        else:
            # ── Release pinch → drop tile ─────────────────────────────────────
            if self.holding is not None:
                # Find the tile that occupies the render-centre of the dragged tile
                cx = int(round(self.holding.render_x)) + TILE_SIZE // 2
                cy = int(round(self.holding.render_y)) + TILE_SIZE // 2
                dest = puzzle.tile_at_pixel(cx, cy)
                if dest and dest is not self.holding:
                    puzzle.swap(self.holding, dest)
                else:
                    self.holding.set_target_grid()  # return home
                self.holding = None

        return self.holding

    def release(self):
        """Called when this hand disappears from the frame."""
        if self.holding:
            self.holding.set_target_grid()
            self.holding = None


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 + 2:  CAMERA PREVIEW  &  CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def run_capture_screen(cap) -> "np.ndarray | None":
    """
    Phase 1 — Live camera preview.

    Shows the webcam feed with on-screen instructions.
    Waits for the user to press SPACE to capture a frame.
    Returns the captured BGR frame, or None if ESC was pressed.

    The camera is NOT closed here; the same 'cap' is reused by the game loop
    for hand tracking.
    """
    print("[CAPTURE SCREEN]  Press SPACE to take puzzle photo.  ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror the frame for a natural selfie view
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = display.shape[:2]

        # ── Semi-transparent instruction banner ───────────────────────────────
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        shadow_text(display, "Position yourself in the frame",
                    (20, h - 100), FONT, 0.65, C_CAPTION, 2)
        shadow_text(display, "Press  SPACE  to capture puzzle image",
                    (20, h -  65), FONT, 0.68, C_TEXT, 2)
        shadow_text(display, "Press  ESC  to quit",
                    (20, h -  30), FONT, 0.55, (150, 150, 200), 1)

        # ── Top label ─────────────────────────────────────────────────────────
        shadow_text(display, "GESTURE PUZZLE  |  Camera Preview",
                    (20, 34), FONT_B, 0.7, C_CAPTION, 2)

        cv2.imshow("Gesture Puzzle  —  Camera Preview", display)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:       # ESC → abort
            cv2.destroyWindow("Gesture Puzzle  —  Camera Preview")
            return None
        elif key == ord(' '):    # SPACE → capture
            cv2.destroyWindow("Gesture Puzzle  —  Camera Preview")
            # Return the raw (pre-display) frame for the puzzle
            return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD  (heads-up display drawn onto the puzzle window)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(canvas: np.ndarray, puzzle: Puzzle, fps: float,
             is_pinching: bool, dragging: bool):
    """
    Draw the HUD overlay at the BOTTOM of the 450 × 450 puzzle canvas.

    Displays:
        • Instructions  ("Pinch to select" / "Release to drop")
        • Move counter
        • Timer
        • FPS
    """
    bar_h = 70
    y0    = PUZZLE_SIZE - bar_h

    # Semi-transparent dark bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (PUZZLE_SIZE, PUZZLE_SIZE), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, canvas, 0.40, 0, canvas)

    # ── Instruction line ──────────────────────────────────────────────────────
    if dragging:
        instruction = "Release to drop"
        inst_col    = (0, 200, 255)
    else:
        instruction = "Pinch to select"
        inst_col    = C_TEXT

    shadow_text(canvas, instruction,
                (10, y0 + 22), FONT, 0.55, inst_col, 1)

    # ── Stats ─────────────────────────────────────────────────────────────────
    elapsed = int(time.time() - puzzle.start_time)
    mm, ss  = divmod(elapsed, 60)

    shadow_text(canvas, f"Moves: {puzzle.move_count}",
                (10, y0 + 44), FONT, 0.5, C_TEXT, 1)
    shadow_text(canvas, f"{mm:02d}:{ss:02d}",
                (155, y0 + 44), FONT, 0.5, C_TEXT, 1)
    shadow_text(canvas, f"FPS: {fps:.1f}",
                (250, y0 + 44), FONT, 0.5, C_FPS, 1)

    # ── Pinch indicator dot ───────────────────────────────────────────────────
    dot_col = C_PINCH_ACTIVE if is_pinching else C_PINCH_IDLE
    cv2.circle(canvas, (PUZZLE_SIZE - 20, y0 + 18), 8, dot_col, -1)
    cv2.circle(canvas, (PUZZLE_SIZE - 20, y0 + 18), 9, (255, 255, 255), 1)
    shadow_text(canvas, "pinch",
                (PUZZLE_SIZE - 58, y0 + 65), FONT, 0.36, dot_col, 1)


def draw_solved_banner(canvas: np.ndarray):
    """
    Draw a large centred 'PUZZLE SOLVED' message with a semi-transparent overlay.
    """
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (PUZZLE_SIZE, PUZZLE_SIZE), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

    banner = "PUZZLE SOLVED"
    (tw, th), _ = cv2.getTextSize(banner, FONT_B, 1.3, 3)
    bx = (PUZZLE_SIZE - tw) // 2
    by = PUZZLE_SIZE // 2 + th // 2 - 20

    # Shadow
    cv2.putText(canvas, banner, (bx + 3, by + 3),
                FONT_B, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
    # Text
    cv2.putText(canvas, banner, (bx, by),
                FONT_B, 1.3, C_SOLVED, 3, cv2.LINE_AA)

    # Sub-line
    elapsed = ""   # already shown in HUD
    sub = "Press R to play again"
    (sw, _), _ = cv2.getTextSize(sub, FONT, 0.55, 1)
    shadow_text(canvas, sub,
                ((PUZZLE_SIZE - sw) // 2, by + 42), FONT, 0.55, C_TEXT, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CURSOR OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def draw_cursors(canvas: np.ndarray,
                 hand_data: list[tuple]):
    """
    Draw a small cursor circle at the pinch midpoint for each detected hand.

    hand_data: list of (distance, mid_x, mid_y) — one entry per hand.
    """
    for dist, mx, my in hand_data:
        is_pinching = dist < PINCH_THRESH
        colour      = C_PINCH_ACTIVE if is_pinching else C_PINCH_IDLE
        radius      = 10 if is_pinching else 14
        cv2.circle(canvas, (mx, my), radius,      colour,          2)
        cv2.circle(canvas, (mx, my), radius // 3, colour,         -1)

        # Show distance number (useful for tuning threshold)
        shadow_text(canvas, f"{int(dist)}",
                    (mx + 16, my - 6), FONT, 0.45, colour, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── MediaPipe Hands ───────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode          = False,
        max_num_hands              = 2,
        min_detection_confidence   = 0.70,
        min_tracking_confidence    = 0.60,
    )

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Exiting.")
        return

    # Higher resolution for hand detection (downscaled later for puzzle)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 + 2:  Camera preview → capture
    # ══════════════════════════════════════════════════════════════════════════
    captured_frame = run_capture_screen(cap)
    if captured_frame is None:
        print("Capture cancelled. Goodbye!")
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3:  Split captured image into tiles
    #  STEP 4:  Shuffle tiles
    #  (Both happen inside Puzzle.__init__)
    # ══════════════════════════════════════════════════════════════════════════
    puzzle = Puzzle(captured_frame)
    print(f"Puzzle created — {GRID_N}×{GRID_N} grid, {len(puzzle.tiles)} tiles.")

    # ── One PinchHand tracker per possible hand ────────────────────────────────
    # max_num_hands = 2, so two trackers are sufficient
    pinch_hands = [PinchHand(), PinchHand()]

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps_display = 0.0
    fps_counter = 0
    fps_timer   = time.time()

    print("Game running — pinch a tile to drag it, release to swap.  ESC = quit.")

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN GAME LOOP
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            # Camera failure — try to continue rather than crash
            print("WARNING: Failed to read camera frame.")
            continue

        # Mirror frame for natural hand interaction
        raw_frame = cv2.flip(raw_frame, 1)

        # ── Hand detection ─────────────────────────────────────────────────────
        # Scale hand landmark coordinates to PUZZLE_SIZE for puzzle interaction
        fh, fw = raw_frame.shape[:2]

        rgb     = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        # Collect per-hand data: (dist, mid_x, mid_y) mapped to puzzle coords
        hand_data: list[tuple] = []   # for cursor drawing

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                dist, mx, my = pinch_distance(hlm, PUZZLE_SIZE, PUZZLE_SIZE)
                hand_data.append((dist, mx, my))
        
        # Pad to always have 2 entries (None = hand not present)
        while len(hand_data) < 2:
            hand_data.append(None)

        # ── Determine which tile is globally being dragged ─────────────────────
        # Only ONE tile can be dragged at a time (first come, first served)
        global_drag: Tile | None = None
        for ph in pinch_hands:
            if ph.holding is not None:
                global_drag = ph.holding
                break

        # ── Process each hand FSM ──────────────────────────────────────────────
        for i, hdata in enumerate(hand_data[:2]):
            if hdata is not None:
                dist, mx, my = hdata
                pinch_hands[i].process(dist, mx, my, puzzle, global_drag)
            else:
                # Hand no longer visible → release any held tile
                pinch_hands[i].release()

        # Re-check global drag after processing
        global_drag = None
        for ph in pinch_hands:
            if ph.holding is not None:
                global_drag = ph.holding
                break

        # ── Puzzle physics update (lerp all tiles) ─────────────────────────────
        puzzle.update(global_drag)

        # ── Solved check ───────────────────────────────────────────────────────
        if not puzzle.solved:
            puzzle.solved = puzzle.check_solved()

        # ══════════════════════════════════════════════════════════════════════
        #  RENDER
        # ══════════════════════════════════════════════════════════════════════

        # Black canvas for the puzzle
        canvas = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE, 3), dtype=np.uint8)

        # 1. Puzzle tiles + grid lines
        puzzle.render(canvas, global_drag)

        # 2. Cursors (thumb-index midpoints)
        valid_hand_data = [hd for hd in hand_data if hd is not None]
        draw_cursors(canvas, valid_hand_data)

        # 3. HUD strip
        any_pinching = any(
            hd is not None and hd[0] < PINCH_THRESH
            for hd in hand_data
        )
        draw_hud(canvas, puzzle, fps_display,
                 is_pinching=any_pinching,
                 dragging=(global_drag is not None))

        # 4. Solved banner
        if puzzle.solved:
            draw_solved_banner(canvas)

        # ── Display ────────────────────────────────────────────────────────────
        cv2.imshow("Gesture Puzzle  —  ESC to quit", canvas)

        # ── FPS calculation ────────────────────────────────────────────────────
        fps_counter += 1
        elapsed_fps = time.time() - fps_timer
        if elapsed_fps >= 1.0:
            fps_display = fps_counter / elapsed_fps
            fps_counter = 0
            fps_timer   = time.time()

        # ── Keyboard input ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:               # ESC → quit
            print("ESC pressed. Exiting.")
            break

        elif key == ord('r'):       # R → reshuffle
            puzzle.shuffle()
            puzzle.solved     = False
            puzzle.move_count = 0
            puzzle.start_time = time.time()
            for ph in pinch_hands:
                ph.release()
            print("Reshuffled!")

    # ── Clean shutdown ─────────────────────────────────────────────────────────
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Game over. Goodbye!")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()