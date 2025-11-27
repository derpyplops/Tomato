"""Integration tests for Modal streaming endpoints.

Run with: python tests/test_modal_streaming.py <modal-base-url>
"""
import sys
import json
import requests


def test_encode_stream(base_url: str):
    """Test that /encode_stream returns valid SSE events."""
    print("Testing /encode_stream...")

    response = requests.post(
        f"{base_url}/encode_stream",
        json={
            "plaintext": "test",
            "prompt": "Write a short story:",
            "chunk_size": 1
        },
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = []
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event = json.loads(line[6:])
                events.append(event)
                print(f"  Event: {event.get('type')} - {str(event)[:60]}...")

    # Verify we got token events and a complete event
    token_events = [e for e in events if e.get('type') == 'token']
    complete_events = [e for e in events if e.get('type') == 'complete']

    assert len(token_events) > 0, "Expected at least one token event"
    assert len(complete_events) == 1, "Expected exactly one complete event"
    assert 'stegotext' in complete_events[0], "Complete event should have stegotext"

    print(f"  PASS: Got {len(token_events)} token events + 1 complete event")
    return complete_events[0]['stegotext']


def test_decode_stream(base_url: str, stegotext: list):
    """Test that /decode_stream returns valid SSE events."""
    print("\nTesting /decode_stream...")

    response = requests.post(
        f"{base_url}/decode_stream",
        json={
            "stegotext": stegotext,
            "prompt": "Write a short story:",
            "chunk_size": 5
        },
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    events = []
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event = json.loads(line[6:])
                events.append(event)
                if event.get('type') == 'token':
                    print(f"  Guess: '{event.get('current_guess', '')[:20]}...' "
                          f"(conf: {event.get('confidence', 0):.2f})")
                else:
                    print(f"  Event: {event.get('type')}")

    # Verify events
    token_events = [e for e in events if e.get('type') == 'token']
    complete_events = [e for e in events if e.get('type') == 'complete']

    assert len(token_events) > 0, "Expected at least one token event"
    assert len(complete_events) == 1, "Expected exactly one complete event"
    assert 'plaintext' in complete_events[0], "Complete event should have plaintext"

    # Verify confidence increases over time
    confidences = [e.get('confidence', 0) for e in token_events]
    if len(confidences) >= 2:
        assert confidences[-1] >= confidences[0], \
            f"Confidence should increase: {confidences[0]} -> {confidences[-1]}"

    print(f"  PASS: Decoded to '{complete_events[0]['plaintext']}'")
    return complete_events[0]['plaintext']


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_modal_streaming.py <modal-base-url>")
        print("Example: python test_modal_streaming.py https://yourname--stego-server-stegoserver")
        sys.exit(1)

    base_url = sys.argv[1].rstrip('/')

    stegotext = test_encode_stream(base_url)
    plaintext = test_decode_stream(base_url, stegotext)

    assert plaintext == "test", f"Roundtrip failed: expected 'test', got '{plaintext}'"
    print(f"\nAll tests passed! Roundtrip successful.")
