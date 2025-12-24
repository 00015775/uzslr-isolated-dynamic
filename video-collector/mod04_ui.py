import os
from mod02_storage import (
    list_signers, add_sign, load_sign_list, recorded_signs,
    print_tree, ensure_folders, count_repetitions
)

# 1. Signer selection / creation
def select_signer() -> str:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_tree()
        print("=== SIGNER MENU ===")
        print("Enter a new signer ID (e.g. signer03) or pick an existing one.")
        print("Type 'q' to quit.\n")
        inp = input("Signer ID: ").strip()
        if inp.lower() == 'q':
            exit(0)
        if not inp.startswith("signer") or not inp[6:].isdigit():
            print("Invalid format – must be signerXX where XX is a number.")
            input("Press Enter to continue...")
            continue
        # create folder if new
        from mod02_storage import path_signer
        path_signer(inp).mkdir(parents=True, exist_ok=True)
        return inp


# 2. Sign-word grid (numbered) + add new
def select_sign(signer_id: str) -> str:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_tree(signer_id)
        signs = load_sign_list(signer_id)
        recorded = recorded_signs(signer_id)

        print("=== SIGN WORD LIST ===")
        COLS = 5
        ROWS = (len(signs) + COLS - 1) // COLS
        for row in range(ROWS):
            line = []
            for col in range(COLS):
                idx = row + col * ROWS
                if idx < len(signs):
                    s = signs[idx]
                    color = "\033[92m" if s in recorded else "\033[97m"
                    line.append(f"{idx+1:3d}. {color}{s.ljust(18)}\033[0m")
                else:
                    line.append(" " * 22)
            print("  ".join(line))
            
        print("\n[a] Add new word")
        print("[b] Back to signer menu")
        print("[q] Quit\n")

        choice = input("Select number, a, b or q: ").strip().lower()

        if choice == 'q':
            exit(0)
        if choice == 'b':
            return None   # signal: go back
        if choice == 'a':
            new = input("New sign word: ").strip()
            if new:
                add_sign(signer_id, new)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(signs):
            return signs[int(choice) - 1]

        print("Invalid selection.")
        input("Press Enter...")


# 3. Repetition menu (shown AFTER each 32-frame capture)
def after_recording_menu(signer_id: str, sign: str, rep_idx: int) -> str:
    """
    Returns:
        "again"  -> record another repetition
        "done"   -> go back to sign list
    """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_tree(signer_id, sign)
        print(f"Finished repetition {rep_idx+1} for sign **{sign}**")
        print("[s] Record another repetition")
        print("[d] Done – back to sign list")
        key = input("Choice: ").strip().lower()
        if key == 's':
            return "again"
        if key == 'd':
            return "done"
        print("Press s or d.")

