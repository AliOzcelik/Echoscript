"""Unpack / pack the UI logic inside Echoscript_standalone.html.

The HTML is a self-contained bundle (React and fonts included, works offline): the page itself is a
JSON string in <script type="__bundler/template">, and the UI logic (class Component) is the
<script type="text/x-dc"> inside that page. Edit the logic as a normal .js file, then pack it back.

    python tools/ui_bundle.py unpack     # Echoscript_standalone.html -> ui_logic.js
    python tools/ui_bundle.py pack       # ui_logic.js -> Echoscript_standalone.html
"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_TAG = '<script type="__bundler/template">'
LOGIC_TAG = '<script type="text/x-dc"'


def read_template(html):
    """-> (page, start, end): the page inside the bundle, and where its JSON sits in the HTML."""
    start = html.index(TEMPLATE_TAG) + len(TEMPLATE_TAG)
    end = html.index("</script>", start)
    return json.loads(html[start:end]), start, end


def logic_span(page):
    start = page.index(">", page.index(LOGIC_TAG)) + 1
    return start, page.index("</script>", start)


def unpack(html_path, js_path):
    page, _, _ = read_template(html_path.read_text(encoding="utf-8"))
    s, e = logic_span(page)
    js_path.write_text(page[s:e], encoding="utf-8")


def pack(html_path, js_path):
    html = html_path.read_text(encoding="utf-8")
    logic = js_path.read_text(encoding="utf-8")
    if "</" in logic:
        raise SystemExit(f"{js_path}: '</' would end the logic's <script> early, write it as '<\\/'")

    page, start, end = read_template(html)
    s, e = logic_span(page)
    page = page[:s] + logic + page[e:]
    # the bundler's own encoding: raw UTF-8, and "</" escaped so the JSON can't close its <script>
    data = "\n" + json.dumps(page, ensure_ascii=False).replace("</", "<\\u002F") + "\n  "
    html_path.write_text(html[:start] + data + html[end:], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Unpack / pack the UI logic of the standalone HTML.")
    ap.add_argument("command", choices=["unpack", "pack"])
    ap.add_argument("--html", type=Path, default=ROOT / "Echoscript_standalone.html")
    ap.add_argument("--js", type=Path, default=ROOT / "ui_logic.js")
    args = ap.parse_args()

    if args.command == "unpack":
        unpack(args.html, args.js)
        print(f"{args.html.name} -> {args.js}")
    else:
        pack(args.html, args.js)
        print(f"{args.js.name} -> {args.html}")


if __name__ == "__main__":
    main()
