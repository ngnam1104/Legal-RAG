import subprocess
import sys
import os
import time

def run_script(script_path):
    print(f"\n\n>>> RUNNING: {script_path}")
    print("-" * 40)
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy()
    )
    process.wait()
    return process.returncode

def main():
    start_time = time.perf_counter()
    
    # 1. Chạy test_chatbot.py (Single-turn / 3-mode)
    code1 = run_script("tests/mode_router/test_chatbot.py")
    
    # 2. Chạy test_chatbot_conversation.py (Multi-turn)
    code2 = run_script("tests/conversation/test_chatbot_conversation.py")
    
    total_duration = time.perf_counter() - start_time
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED.")
    print(f"Total Execution Time: {total_duration:.2f}s")
    print(f"Exit Codes: test_chatbot={code1}, test_conversation={code2}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
