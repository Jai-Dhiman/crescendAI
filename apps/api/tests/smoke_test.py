#!/usr/bin/env python3
"""
Smoke test for CrescendAI API.
Run against local dev server (just api) at http://localhost:8787.

Usage:
    uv run python tests/smoke_test.py              # run tests
    uv run python tests/smoke_test.py --baseline    # save baseline
    uv run python tests/smoke_test.py --compare     # compare to baseline
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://localhost:8787"
BASELINE_FILE = Path(__file__).parent / "smoke_baseline.json"

UNAUTH_TESTS = [
    ("GET", "/health", None, 200, {"has_key": "status"}),
    ("GET", "/api/auth/me", None, 401, {"has_key": "error"}),
    ("POST", "/api/auth/apple", {"identityToken": "invalid", "userId": "test"}, [400, 401], None),
    ("POST", "/api/ask", {}, 401, None),
    ("POST", "/api/sync", {}, 401, None),
    ("POST", "/api/waitlist", {"email": "smoke@test.com"}, [200, 400, 409], None),
    ("GET", "/api/nonexistent", None, [404, 405], None),
]


def make_request(method, path, body=None, headers=None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req)
        body_bytes = resp.read()
        try:
            body_json = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        return resp.status, body_json, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            body_json = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        return e.code, body_json, dict(e.headers)


def run_unauth_tests():
    results = []
    for method, path, body, expected, checks in UNAUTH_TESTS:
        status, resp_body, resp_headers = make_request(method, path, body)
        expected_list = expected if isinstance(expected, list) else [expected]
        passed = status in expected_list
        if passed and checks:
            if "has_key" in checks:
                passed = checks["has_key"] in resp_body
            if "header" in checks:
                passed = passed and checks["header"].lower() in {
                    k.lower() for k in resp_headers
                }
        result = {
            "method": method,
            "path": path,
            "status": status,
            "expected": expected_list,
            "passed": passed,
        }
        results.append(result)
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {method} {path} -> {status} (expected {expected_list})")
    return results


def run_auth_flow():
    results = []
    status, body, headers = make_request(
        "POST", "/api/auth/debug", {"email": "smoke@test.com"}
    )
    if status == 404:
        print("  [SKIP] Debug auth not available (production mode)")
        return results

    passed = status == 200 and "student_id" in body
    results.append({"test": "debug_login", "status": status, "passed": passed})
    print(f"  [{'PASS' if passed else 'FAIL'}] POST /api/auth/debug -> {status}")
    if not passed:
        return results

    token = body.get("token", "")
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    status, body, _ = make_request("GET", "/api/auth/me", headers=auth_headers)
    passed = status == 200
    results.append({"test": "auth_me", "status": status, "passed": passed})
    print(f"  [{'PASS' if passed else 'FAIL'}] GET /api/auth/me -> {status}")

    status, body, _ = make_request("GET", "/api/conversations", headers=auth_headers)
    passed = status == 200
    results.append({"test": "list_conversations", "status": status, "passed": passed})
    print(f"  [{'PASS' if passed else 'FAIL'}] GET /api/conversations -> {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="API Smoke Test")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare to saved baseline")
    args = parser.parse_args()

    print("\n=== CrescendAI API Smoke Test ===\n")

    print("Unauthenticated tests:")
    unauth_results = run_unauth_tests()

    print("\nAuthenticated flow:")
    auth_results = run_auth_flow()

    all_results = {"unauth": unauth_results, "auth": auth_results}

    total = len(unauth_results) + len(auth_results)
    passed = sum(1 for r in unauth_results if r["passed"]) + sum(
        1 for r in auth_results if r["passed"]
    )
    print(f"\n--- {passed}/{total} tests passed ---\n")

    if args.baseline:
        BASELINE_FILE.write_text(json.dumps(all_results, indent=2))
        print(f"Baseline saved to {BASELINE_FILE}")

    if args.compare and BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
        diffs = []
        for test_type in ["unauth", "auth"]:
            for i, result in enumerate(all_results.get(test_type, [])):
                if i < len(baseline.get(test_type, [])):
                    base = baseline[test_type][i]
                    if result["status"] != base["status"]:
                        diffs.append(
                            f"  {test_type}[{i}]: {base['status']} -> {result['status']}"
                        )
        if diffs:
            print("REGRESSIONS:")
            for d in diffs:
                print(d)
            sys.exit(1)
        else:
            print("No regressions detected vs baseline.")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
