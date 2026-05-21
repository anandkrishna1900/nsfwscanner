import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
transcript_path = r"C:\Users\anand\.gemini\antigravity\brain\d9da3e4f-2322-45c7-b560-7f6e081aa356\.system_generated\logs\transcript.jsonl"

for line in open(transcript_path, encoding="utf-8"):
    obj = json.loads(line)
    step = obj.get('step_index', 0)
    if "test_hentai.png" in line and 968 < step < 1040:
        print(f"=== Step {step}: type={obj.get('type')}, source={obj.get('source')} ===")
        if obj.get("tool_calls"):
            print("Tool calls:", json.dumps(obj.get("tool_calls"), indent=2))
        if obj.get("content"):
            print("Content snippet:", obj.get("content")[:1000])
        print()
