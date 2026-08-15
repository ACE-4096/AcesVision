"""Local preview server consumed by the future Qt/QML client."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>AcesVision Preview</title>
<style>body{margin:0;background:#101216;color:#eee;font:14px sans-serif}
main{max-width:1100px;margin:auto;padding:16px}img{width:100%;background:#000}
#state{padding:8px 0;color:#aaa}</style></head><body><main>
<h1>AcesVision</h1><div id=state>connecting</div><img id=feed>
<script>const image=document.getElementById('feed'),state=document.getElementById('state');
setInterval(()=>{image.src='/latest.jpg?t='+Date.now()},100);
setInterval(async()=>{try{const r=await fetch('/api/state');const s=await r.json();
state.textContent=s.status+' | '+s.source+' | frame '+s.sequence}catch(e){}},500);
</script></main></body></html>"""


class PreviewServer:
    def __init__(self, latest_output, pipeline, host="127.0.0.1", port=8765):
        self.latest_output = latest_output
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index.html"):
                    self._reply(200, "text/html; charset=utf-8", PAGE)
                    return
                if self.path.startswith("/latest.jpg"):
                    _, jpeg = outer.latest_output.snapshot()
                    if not jpeg:
                        self._reply(503, "text/plain", b"camera unavailable")
                    else:
                        self._reply(200, "image/jpeg", jpeg,
                                    {"Cache-Control": "no-store"})
                    return
                if self.path == "/api/state":
                    state = outer.pipeline.state()
                    body = json.dumps({
                        "status": state.status,
                        "source": state.source.safe_label(),
                        "sequence": state.sequence,
                        "last_error": state.last_error,
                        "metrics": state.metrics,
                    }).encode()
                    self._reply(200, "application/json", body,
                                {"Cache-Control": "no-store"})
                    return
                self._reply(404, "text/plain", b"not found")

            def _reply(self, status, content_type, body, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True, name="vision-preview")
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
