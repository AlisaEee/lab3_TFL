import http.server
import socketserver
import json
from typing import Union, Dict, Any
from http import HTTPStatus
# Assume tokenize, parse_CFG, and chomskiy_normalize are defined elsewhere
from test import Grammar, read_grammar_from_file,generateTests, read_grammar,eliminate_chain_rules,convert

def generate_Tests(n_tests: int) -> list[str]:
    grammar_string = read_grammar_from_file("grammar.txt")
    grammar = read_grammar(grammar_string)
    grammar.set_start_symbol('S')
    new_grammar = eliminate_chain_rules(grammar)
    new_grammar = convert(new_grammar)
    new_grammar.set_start_symbol('S')
    ans = generateTests(new_grammar, n_tests)
    
    return ans


def request_handler(request: http.server.BaseHTTPRequestHandler):
    global cfg
    try:
        if request.command == "POST":
            content_length = int(request.headers['Content-Length'])
            body_str = request.rfile.read(content_length).decode('utf-8')
            body = json.loads(body_str)

            if request.path == "/getTests":
                
                response = generate_Tests(body["n_tests"])
                request.send_response(HTTPStatus.OK)
                request.send_header("Content-Type", "application/json")
                request.end_headers()
                request.wfile.write(json.dumps(response).encode('utf-8'))
                return
            else:
                request.send_response(HTTPStatus.NOT_FOUND)
                request.send_header("Content-Type", "application/json")
                request.end_headers()
                request.wfile.write(json.dumps({"error": "Invalid endpoint"}).encode('utf-8'))
                return
        else:
            request.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            request.send_header("Content-Type", "application/json")
            request.end_headers()
            request.wfile.write(json.dumps({"error": "Method not allowed"}).encode('utf-8'))
            return
    except Exception as e:
        request.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        request.send_header("Content-Type", "application/json")
        request.end_headers()
        request.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        return


class MyServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.allow_reuse_address = True


class MyRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
      request_handler(self)

    def do_GET(self):
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error":"GET method not allowed"}).encode('utf-8'))
        return


if __name__ == "__main__":
    port = 8081
    print(f"Starting server on port {port}...")
    with MyServer(("", port), MyRequestHandler) as httpd:
        httpd.serve_forever()