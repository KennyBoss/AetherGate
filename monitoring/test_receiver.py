"""CI-only, local webhook sink. Never used by the default monitoring stack."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


notifications = []
lock = threading.Lock()


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if self.path != "/alerts" or not 0 < size <= 1024 * 1024:
            self.send_error(400)
            return
        try:
            notification = json.loads(self.rfile.read(size))
        except ValueError:
            self.send_error(400)
            return
        with lock:
            notifications.append(notification)
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path != "/notifications":
            self.send_error(404)
            return
        with lock:
            payload = json.dumps(notifications).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Receiver).serve_forever()
