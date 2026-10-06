"""Processus de test du protocole FLM : aucune IA ni téléchargement."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    arguments = sys.argv[1:]
    command = arguments[0]
    if command == "validate":
        print("NPU simulé : validation réussie", flush=True)
        return
    if command == "pull":
        print("Téléchargement simulé : 100% · 2 Mio/s", flush=True)
        return
    if command != "serve":
        sys.exit(3)
    model = arguments[1]
    port = int(arguments[arguments.index("--port") + 1])
    requests = {}
    last_payload = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/ps":
                self.reply({"models": [{"name": model}]})
            elif self.path == "/test/payload":
                self.reply(last_payload)
            else:
                self.reply({"error": "Unknown endpoint"}, 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/cancel":
                event = requests.get(payload["request_id"])
                if event:
                    event.set()
                self.reply({"cancelled": event is not None})
                return
            if self.path != "/v1/chat/completions":
                self.reply({"error": "Unknown endpoint"}, 404)
                return
            last_payload.clear()
            last_payload.update(payload)
            prompt = payload["messages"][0]["content"][0]["text"]
            if prompt == "FAIL":
                self.reply({"error": "No memory"}, 500)
                return
            if prompt == "REDIRECT":
                self.send_response(307)
                self.send_header("Location", "https://example.com/webcam")
                self.end_headers()
                return
            event = threading.Event()
            requests[payload["request_id"]] = event
            try:
                if prompt == "WAIT":
                    if not event.wait(5):
                        self.reply({"error": "Cancellation timeout"}, 500)
                        return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for text in ("Un carré ", "rouge."):
                    chunk = {"choices": [{"delta": {"content": text}}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                requests.pop(payload["request_id"], None)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
