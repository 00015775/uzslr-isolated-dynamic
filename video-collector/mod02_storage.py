import os
import json
from pathlib import Path
from mod01_config import DATA_ROOT, DEFAULT_SIGNS
from typing import Optional

# Helper: make sure the whole dataset root exists
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)


# 1. Paths for a given signer -> sign -> repetition
def path_signer(signer_id: str) -> Path:
    return Path(DATA_ROOT) / signer_id

def path_sign(signer_id: str, sign: str) -> Path:
    return path_signer(signer_id) / sign

def path_videos(signer_id: str, sign: str) -> Path:
    return path_sign(signer_id, sign) / "videos"

def path_landmarks(signer_id: str, sign: str) -> Path:
    return path_sign(signer_id, sign) / "landmarks"


# 2. Create all folders for a (signer, sign) pair
def ensure_folders(signer_id: str, sign: str):
    for p in (path_videos(signer_id, sign), path_landmarks(signer_id, sign)):
        p.mkdir(parents=True, exist_ok=True)


# 3. GLOBAL sign-word list (shared by EVERY signer)
def _meta_path() -> Path:
    """One single meta file in the dataset root."""
    return Path(DATA_ROOT) / "meta.json"

def load_sign_list(signer_id: str) -> list[str]:
    """Return the *global* list – ignore the signer_id."""
    p = _meta_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # makes sure that newly added signs are shown across all signers
            return data.get("global", DEFAULT_SIGNS[:])
        except Exception:
            pass
    return DEFAULT_SIGNS[:]

def save_sign_list(signer_id: str, signs: list[str]):
    p = _meta_path()
    # keep any old signs
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    else:
        data = {}
    data["global"] = signs
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def add_sign(signer_id: str, new_sign: str):
    signs = load_sign_list(signer_id)      # still works, returns global list
    if new_sign not in signs:
        signs.append(new_sign)
        save_sign_list(signer_id, signs)  # signer_id is ignored inside


# 4. Progress: how many repetitions already exist for a sign?
def count_repetitions(signer_id: str, sign: str) -> int:
    vid_dir = path_videos(signer_id, sign)
    if not vid_dir.exists():
        return 0
    # each repetition -> a sub-folder "rep-0", "rep-1", …
    return len([d for d in vid_dir.iterdir() if d.is_dir() and d.name.startswith("rep-")])


# 5. Which signs have AT LEAST ONE repetition?
def recorded_signs(signer_id: str) -> set[str]:
    signs = load_sign_list(signer_id)
    recorded = set()
    for s in signs:
        if count_repetitions(signer_id, s) > 0:
            recorded.add(s)
    return recorded


# 6. List ALL signers that already exist
def list_signers() -> list[str]:
    return [d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir() and d.name.startswith("signer")]


# 7. Pretty tree view – now with BLUE highlighting for the active node
ANSI_BLUE   = "\033[94m"
ANSI_GREEN  = "\033[92m"
ANSI_WHITE  = "\033[97m"
ANSI_RESET  = "\033[0m"

def print_tree(signer_id: Optional[str] = None,
               current_sign: Optional[str] = None,
               current_rep: Optional[int] = None):
    """
    Parameters
    ----------
    signer_id   – the signer we are currently inside (or None)
    current_sign – the sign we are recording (or None)
    current_rep  – the repetition we are about to record (0-based)
    """
    print("\n=== CURRENT DATASET TREE ===")
    for signer_dir in sorted(Path(DATA_ROOT).iterdir()):
        if not signer_dir.is_dir() or not signer_dir.name.startswith("signer"):
            continue
        sid = signer_dir.name
        recorded = len(recorded_signs(sid))
        total    = len(load_sign_list(sid))

        # signer line
        marker = "→ " if signer_id == sid else "  "
        line = f"{marker}{sid}  [{recorded}/{total} signs recorded]"

        # highlight the ACTIVE signer in BLUE
        if signer_id == sid:
            line = f"{ANSI_BLUE}{line}{ANSI_RESET}"
        print(line)

        # sign list (only for the active signer)
        if signer_id == sid:
            for sign in load_sign_list(sid):
                reps = count_repetitions(sid, sign)
                rec_mark = "✓" if reps > 0 else " "
                cur_mark = " →" if sign == current_sign else "  "

                # base sign line
                sign_line = f"   {cur_mark} [{rec_mark}] {sign}  (reps: {reps})"

                # highlight the ACTIVE sign in BLUE
                if sign == current_sign:
                    sign_line = f"{ANSI_BLUE}{sign_line}{ANSI_RESET}"

                    # add the NEXT repetition number in BLUE as well
                    next_rep = reps   # count_repetitions gives how many exist
                    rep_text = f"  → next rep: {next_rep + 1}"
                    if current_rep is not None:
                        rep_text = f"  → rep: {current_rep + 1}"
                    sign_line += f"{ANSI_BLUE}{rep_text}{ANSI_RESET}"
                else:
                    # recorded signs stay GREEN, unrecorded stay WHITE
                    color = ANSI_GREEN if reps > 0 else ANSI_WHITE
                    sign_line = f"{color}{sign_line}{ANSI_RESET}"

                print(sign_line)
    print("==========================\n")


