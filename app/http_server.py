"""轻量 HTTP 服务入口，用于对外暴露文档转表格能力。"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
import atexit
from urllib.parse import urlparse
import os

import requests

from core.processor import DocumentProcessor
from core.settings import Settings


_cfg = Settings.from_env()
_processor = DocumentProcessor()


def _sanitize_json(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        if self.path == "/upload":
            try:
                content_type = self.headers.get("Content-Type")
                if not content_type:
                    raise ValueError("缺少 Content-Type")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
                )

                filename = None
                file_bytes = None
                if "file" in form:
                    file_item = form["file"]
                    filename = getattr(file_item, "filename", None)
                    file_bytes = file_item.file.read() if getattr(file_item, "file", None) else None

                url = form.getfirst("url")
                excel_path = form.getfirst("excel_path")
                if (not file_bytes) and url:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    file_bytes = resp.content
                    filename = urlparse(url).path.split("/")[-1] or filename

                if not file_bytes:
                    raise ValueError("缺少文件或URL")

                ext = (os.path.splitext(filename)[1] if filename else "").lower()
                if ext == ".pdf":
                    result = _processor.process_pdf_bytes(
                        file_bytes, filename=filename or "input.pdf", excel_output_path=excel_path
                    )
                else:
                    result = _processor.process_image_bytes(
                        file_bytes, filename=filename or "input.png", excel_output_path=excel_path
                    )

                body = json.dumps(_sanitize_json(result), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Endpoint not found")








def run_server(port=None):
    port = int(port or _cfg.lin_port)


    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Starting HTTP server on port {port}...")

    httpd.serve_forever()


def main(port=None):
    run_server(port=port)


if __name__ == "__main__":
    main()
