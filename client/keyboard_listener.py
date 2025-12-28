import socket
import os
import keyboard
import sys
import time
import argparse

SOCKET_PATH = "/tmp/glm_asr_keyboard.sock"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotkey", type=str, default="f12")
    args = parser.parse_args()
    hotkey = args.hotkey

    if os.geteuid() != 0:
        print("Keyboard listener must be run as root (sudo).")
        sys.exit(1)

    print(f"Keyboard listener started. Listening for {hotkey}...")

    def send_event(event_type):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(SOCKET_PATH)
                s.sendall(event_type.encode())
        except Exception:
            # Silently fail if main.py is not listening yet
            pass

    def on_hotkey(e):
        if e.event_type == keyboard.KEY_DOWN:
            send_event("DOWN")
        elif e.event_type == keyboard.KEY_UP:
            send_event("UP")

    keyboard.hook_key(hotkey, on_hotkey)
    
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, EOFError, OSError):
        pass
    finally:
        try:
            keyboard.unhook_all()
        except:
            pass

if __name__ == "__main__":
    main()
