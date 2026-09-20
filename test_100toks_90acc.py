import requests
import time
import json

URL = "http://127.0.0.1:8000/v1/chat/completions"

HIGH_ACCEPTANCE_PROMPTS = [
    {
        "name": "structured_json_schema",
        "system": "You are a database system that outputs strictly valid JSON arrays of objects with fields: id (integer), name (string), role (string), and active (boolean).",
        "prompt": "List 10 employees in the engineering department with realistic names and roles. Output JSON only.",
        "max_tokens": 200,
        "k_target": 7
    },
    {
        "name": "python_dataclass_boilerplate",
        "system": "You are an expert Python systems programmer. Write clean Python code with type annotations.",
        "prompt": "Write a Python dataclass definition for `NetworkPacket` with fields: header_id (int), src_ip (str), dest_ip (str), payload (bytes), timestamp (float), and checksum (int). Include standard __repr__ and validation methods.",
        "max_tokens": 150,
        "k_target": 7
    },
    {
        "name": "counting_and_sequence",
        "system": "You are a precise mathematical assistant.",
        "prompt": "Write a Python loop that counts from 1 to 50, printing each number squared in the format: 'Number X squared is Y'.",
        "max_tokens": 120,
        "k_target": 7
    }
]

print("==========================================================================")
print(" SPARK-SPLASH: Pushing for 100+ tok/s and 90%+ Speculative Acceptance")
print(" Hardware: NVIDIA GB10 (Blackwell sm_121, 128GB LPDDR5X)")
print(" Drafter: DFlash 2 / DSpark v2 with Dynamic Jev K-Selection")
print("==========================================================================\n")

results = []

for item in HIGH_ACCEPTANCE_PROMPTS:
    payload = {
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["prompt"]}
        ],
        "max_tokens": item["max_tokens"],
        "temperature": 0.0,
        "stream": False
    }

    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload)
    wall_time = time.perf_counter() - t0

    if resp.status_code == 200:
        data = resp.json()
        timings = data.get("timings", {})
        meta = data.get("spark_splash_meta", {})
        
        gen_tokens = data["usage"]["completion_tokens"]
        toks_sec = timings.get("predicted_per_second", gen_tokens / (timings.get("predicted_ms", 1000) / 1000.0))
        
        draft_n = timings.get("draft_n", 0)
        draft_acc = timings.get("draft_n_accepted", 0)
        acc_rate = (draft_acc / draft_n * 100.0) if draft_n else 0.0
        
        print(f"Workload: {item['name']}")
        print(f"  - Generated Tokens: {gen_tokens}")
        print(f"  - Draft Tokens: {draft_n}, Accepted: {draft_acc}")
        print(f"  - Acceptance Rate: {acc_rate:.2f}%")
        print(f"  - Decode Speed (Server): {toks_sec:.2f} tok/s")
        print(f"  - Jev Dynamic K: K={meta.get('k_speculation')} (Conf: {meta.get('jev_confidence')})")
        print(f"  - Total Wall Time: {wall_time:.2f}s\n")
        
        results.append({
            "name": item["name"],
            "acceptance_rate": acc_rate,
            "toks_sec": toks_sec,
            "tokens": gen_tokens
        })
    else:
        print(f"Failed {item['name']}: {resp.status_code} {resp.text}")

print("==========================================================================")
print("BENCHMARK SUMMARY:")
for r in results:
    status_90 = "PASS (>=90%)" if r["acceptance_rate"] >= 90.0 else f"{r['acceptance_rate']:.1f}%"
    print(f"  • {r['name']:<30} | Acceptance: {status_90:<12} | Decode: {r['toks_sec']:.2f} tok/s")
print("==========================================================================")
