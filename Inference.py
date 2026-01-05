import argparse
import json
import random
import time
from typing import Dict, Any

import requests

JOB_TYPES = ["employee", "entrepreneur", "freelance", "civil_servant"]
EDU_LEVELS = ["high_school", "diploma", "bachelor", "master"]
CITIES = ["Bekasi", "Jakarta", "Bandung", "Surabaya", "Yogyakarta"]
MARITAL = ["single", "married", "divorced"]

def make_record() -> Dict[str, Any]:
    return {
        "age": random.randint(21, 60),
        "monthly_income": random.randint(3_000_000, 35_000_000),
        "loan_amount": random.randint(1_000_000, 80_000_000),
        "tenure_months": random.randint(6, 72),
        "num_credit_lines": random.randint(0, 10),
        "job_type": random.choice(JOB_TYPES),
        "education_level": random.choice(EDU_LEVELS),
        "city": random.choice(CITIES),
        "marital_status": random.choice(MARITAL),
        "has_previous_default": random.choice([0, 1]),
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exporter-url", default="http://127.0.0.1:8000/invocations")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--sleep", type=float, default=0.1)
    args = p.parse_args()

    ok = 0
    fail = 0
    for i in range(args.n):
        payload = {"dataframe_records": [make_record()]}
        try:
            r = requests.post(args.exporter_url, json=payload, timeout=15)
            if r.ok:
                ok += 1
            else:
                fail += 1
            print(f"[{i+1}/{args.n}] status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{args.n}] EXCEPTION: {e}")
        time.sleep(args.sleep)

    print(json.dumps({"ok": ok, "fail": fail}, indent=2))

if __name__ == "__main__":
    main()
