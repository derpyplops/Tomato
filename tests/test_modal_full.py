"""
Comprehensive integration tests for Modal server.

Tests all endpoints including streaming, with roundtrip verification.

Usage:
    python tests/test_modal_full.py <modal-base-url>

Example:
    python tests/test_modal_full.py https://yourname--stego-server-stegoserver
"""
import sys
import json
import time
import requests
from typing import List, Dict, Any


class ModalServerTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results: List[Dict[str, Any]] = []

    def test(self, name: str, func):
        """Run a test and record results."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)

        start = time.time()
        try:
            result = func()
            elapsed = time.time() - start
            print(f"PASS ({elapsed:.2f}s)")
            self.results.append({
                'name': name,
                'status': 'PASS',
                'time': elapsed,
                'result': result
            })
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAIL ({elapsed:.2f}s): {e}")
            self.results.append({
                'name': name,
                'status': 'FAIL',
                'time': elapsed,
                'error': str(e)
            })
            return None

    def test_health(self):
        """Test /health endpoint."""
        resp = requests.get(f"{self.base_url}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'healthy'
        print(f"  Model: {data.get('model')}")
        print(f"  GPU: {data.get('gpu')}")
        return data

    def test_encode(self, plaintext: str = "hello", prompt: str = "Write a story:"):
        """Test /encode endpoint."""
        resp = requests.post(
            f"{self.base_url}/encode",
            json={"plaintext": plaintext, "prompt": prompt}
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert 'stegotext' in data
        assert 'formatted_stegotext' in data
        assert len(data['stegotext']) > 0
        print(f"  Tokens: {len(data['stegotext'])}")
        print(f"  Text: {data['formatted_stegotext'][:80]}...")
        return data

    def test_decode(self, stegotext: list, prompt: str = "Write a story:"):
        """Test /decode endpoint."""
        resp = requests.post(
            f"{self.base_url}/decode",
            json={"stegotext": stegotext, "prompt": prompt}
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert 'plaintext' in data
        print(f"  Decoded: '{data['plaintext']}'")
        return data

    def test_encode_stream(self, plaintext: str = "test", prompt: str = "Write a story:"):
        """Test /encode_stream endpoint."""
        resp = requests.post(
            f"{self.base_url}/encode_stream",
            json={"plaintext": plaintext, "prompt": prompt, "chunk_size": 1},
            stream=True
        )
        assert resp.status_code == 200

        events = []
        for line in resp.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                event = json.loads(line.decode('utf-8')[6:])
                events.append(event)

        token_events = [e for e in events if e.get('type') == 'token']
        complete_events = [e for e in events if e.get('type') == 'complete']

        assert len(complete_events) == 1
        print(f"  Token events: {len(token_events)}")
        print(f"  Total tokens: {complete_events[0]['total_tokens']}")
        return complete_events[0]

    def test_decode_stream(self, stegotext: list, prompt: str = "Write a story:"):
        """Test /decode_stream endpoint."""
        resp = requests.post(
            f"{self.base_url}/decode_stream",
            json={"stegotext": stegotext, "prompt": prompt, "chunk_size": 5},
            stream=True
        )
        assert resp.status_code == 200

        events = []
        for line in resp.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                event = json.loads(line.decode('utf-8')[6:])
                events.append(event)
                if event.get('type') == 'token':
                    conf = event.get('confidence', 0)
                    guess = event.get('current_guess', '')[:15]
                    print(f"  [{event['tokens_processed']:3d}] {conf:.2f} '{guess}...'")

        complete_events = [e for e in events if e.get('type') == 'complete']
        assert len(complete_events) == 1
        print(f"  Final: '{complete_events[0]['plaintext']}'")
        return complete_events[0]

    def test_roundtrip(self, plaintext: str = "secret"):
        """Test full encode -> decode roundtrip."""
        prompt = "Write a short story about a magical forest:"

        # Encode
        encode_resp = requests.post(
            f"{self.base_url}/encode",
            json={"plaintext": plaintext, "prompt": prompt}
        )
        assert encode_resp.status_code == 200
        stegotext = encode_resp.json()['stegotext']

        # Decode
        decode_resp = requests.post(
            f"{self.base_url}/decode",
            json={"stegotext": stegotext, "prompt": prompt}
        )
        assert decode_resp.status_code == 200
        decoded = decode_resp.json()['plaintext']

        print(f"  Original:  '{plaintext}'")
        print(f"  Decoded:   '{decoded}'")
        assert decoded == plaintext, f"Roundtrip failed: '{plaintext}' -> '{decoded}'"
        return {'original': plaintext, 'decoded': decoded, 'match': True}

    def test_roundtrip_streaming(self, plaintext: str = "hi"):
        """Test full streaming encode -> streaming decode roundtrip."""
        prompt = "Tell me about space exploration:"

        # Streaming encode
        encode_resp = requests.post(
            f"{self.base_url}/encode_stream",
            json={"plaintext": plaintext, "prompt": prompt, "chunk_size": 1},
            stream=True
        )

        stegotext = None
        for line in encode_resp.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                event = json.loads(line.decode('utf-8')[6:])
                if event.get('type') == 'complete':
                    stegotext = event['stegotext']

        assert stegotext is not None
        print(f"  Encoded {len(stegotext)} tokens")

        # Streaming decode
        decode_resp = requests.post(
            f"{self.base_url}/decode_stream",
            json={"stegotext": stegotext, "prompt": prompt, "chunk_size": 10},
            stream=True
        )

        decoded = None
        for line in decode_resp.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                event = json.loads(line.decode('utf-8')[6:])
                if event.get('type') == 'complete':
                    decoded = event['plaintext']

        print(f"  Original:  '{plaintext}'")
        print(f"  Decoded:   '{decoded}'")
        assert decoded == plaintext, f"Streaming roundtrip failed"
        return {'original': plaintext, 'decoded': decoded, 'match': True}

    def run_all(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("MODAL SERVER INTEGRATION TESTS")
        print(f"Target: {self.base_url}")
        print("="*60)

        # Basic tests
        self.test("Health Check", self.test_health)

        # Non-streaming
        encode_result = self.test(
            "Encode (non-streaming)",
            lambda: self.test_encode("hello", "Write a story:")
        )

        if encode_result:
            self.test(
                "Decode (non-streaming)",
                lambda: self.test_decode(encode_result['stegotext'], "Write a story:")
            )

        # Streaming
        stream_encode_result = self.test(
            "Encode (streaming)",
            lambda: self.test_encode_stream("test", "Write a poem:")
        )

        if stream_encode_result:
            self.test(
                "Decode (streaming)",
                lambda: self.test_decode_stream(
                    stream_encode_result['stegotext'],
                    "Write a poem:"
                )
            )

        # Roundtrips
        self.test("Roundtrip (non-streaming)", lambda: self.test_roundtrip("secret"))
        self.test("Roundtrip (streaming)", lambda: self.test_roundtrip_streaming("hi"))

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')

        for r in self.results:
            status = "PASS" if r['status'] == 'PASS' else "FAIL"
            print(f"  [{status}] {r['name']} ({r['time']:.2f}s)")

        print(f"\nTotal: {passed} passed, {failed} failed")
        return failed == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_modal_full.py <modal-base-url>")
        print("Example: python test_modal_full.py https://yourname--stego-server-stegoserver")
        sys.exit(1)

    tester = ModalServerTester(sys.argv[1])
    success = tester.run_all()
    sys.exit(0 if success else 1)
