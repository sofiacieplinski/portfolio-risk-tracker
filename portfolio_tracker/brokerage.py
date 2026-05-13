"""
Plaid brokerage integration — connect any brokerage and auto-sync holdings.

Supports: Merrill Lynch, Schwab, Fidelity, TD Ameritrade, Robinhood, E*Trade,
          Vanguard, and 10,000+ other institutions via Plaid.

Setup:
  1. Create a free Plaid developer account at https://dashboard.plaid.com/signup
  2. Create an app, enable "Investments" product
  3. Copy your Client ID and Sandbox/Development Secret
  4. Run:
       portfolio-tracker set-key plaid-client-id  <your-client-id>
       portfolio-tracker set-key plaid-secret      <your-secret>
       portfolio-tracker set-key plaid-env         sandbox   # or development / production
  5. portfolio-tracker connect
  6. portfolio-tracker sync
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import requests

from .config import get_key, set_key

PLAID_HOSTS = {
    "sandbox":     "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production":  "https://production.plaid.com",
}

CALLBACK_PORT = 8765


# ─── Plaid API helpers ────────────────────────────────────────────────────────

def _plaid_post(endpoint: str, payload: dict) -> dict:
    env     = get_key("plaid-env") or "sandbox"
    host    = PLAID_HOSTS.get(env, PLAID_HOSTS["sandbox"])
    cid     = get_key("plaid-client-id")
    secret  = get_key("plaid-secret")

    if not cid or not secret:
        raise RuntimeError(
            "Plaid credentials not configured.\n"
            "Run:\n"
            "  portfolio-tracker set-key plaid-client-id  <your-client-id>\n"
            "  portfolio-tracker set-key plaid-secret      <your-secret>\n"
            "  portfolio-tracker set-key plaid-env         sandbox\n"
            "Get your keys at https://dashboard.plaid.com"
        )

    resp = requests.post(
        f"{host}{endpoint}",
        json={"client_id": cid, "secret": secret, **payload},
        timeout=15,
    )
    data = resp.json()
    if "error_code" in data:
        raise RuntimeError(f"Plaid error {data['error_code']}: {data.get('error_message', '')}")
    return data


def _create_link_token(user_id: str = "portfolio-tracker-user") -> str:
    data = _plaid_post("/link/token/create", {
        "user": {"client_user_id": user_id},
        "client_name": "Portfolio Risk Tracker",
        "products": ["investments"],
        "country_codes": ["US"],
        "language": "en",
    })
    return data["link_token"]


def _exchange_public_token(public_token: str) -> str:
    data = _plaid_post("/item/public_token/exchange", {"public_token": public_token})
    return data["access_token"]


# ─── Local OAuth callback server ──────────────────────────────────────────────

_PLAID_LINK_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Connect Your Brokerage</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0d1117; color: #e6edf3;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh;
    }}
    .card {{
      background: #161b22; border: 1px solid #30363d; border-radius: 12px;
      padding: 40px; max-width: 440px; width: 90%; text-align: center;
    }}
    h1 {{ font-size: 22px; margin-bottom: 8px; }}
    p  {{ color: #8b949e; font-size: 14px; margin-bottom: 28px; line-height: 1.6; }}
    button {{
      background: #238636; color: #fff; border: none; border-radius: 6px;
      padding: 12px 28px; font-size: 15px; cursor: pointer; width: 100%;
    }}
    button:hover {{ background: #2ea043; }}
    #status {{ margin-top: 20px; font-size: 14px; color: #8b949e; }}
  </style>
</head>
<body>
<div class="card">
  <h1>Connect Your Brokerage</h1>
  <p>Link your account securely via Plaid.<br>
     Supports Merrill Lynch, Schwab, Fidelity, and 10,000+ institutions.</p>
  <button id="btn">Connect Account</button>
  <div id="status"></div>
</div>
<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
<script>
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');

  const handler = Plaid.create({{
    token: '{link_token}',
    onSuccess: async (public_token, metadata) => {{
      btn.disabled = true;
      status.textContent = 'Exchanging token…';
      const resp = await fetch('/callback', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ public_token, institution: metadata.institution }})
      }});
      if (resp.ok) {{
        status.textContent = '✓ Connected! You can close this tab.';
        document.querySelector('.card h1').textContent = 'Brokerage Connected';
        document.querySelector('.card p').textContent =
          'Run: portfolio-tracker sync   to import your holdings.';
        btn.style.display = 'none';
      }} else {{
        status.textContent = 'Error — check terminal for details.';
      }}
    }},
    onExit: (err) => {{
      if (err) status.textContent = 'Error: ' + err.error_message;
      else     status.textContent = 'Cancelled.';
    }}
  }});

  btn.addEventListener('click', () => handler.open());
</script>
</body>
</html>"""


class _CallbackHandler(BaseHTTPRequestHandler):
    public_token: str | None = None
    institution:  str | None = None

    def do_GET(self) -> None:  # noqa: N802
        html = _PLAID_LINK_HTML.format(link_token=self.server.link_token)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        _CallbackHandler.public_token = body.get("public_token")
        _CallbackHandler.institution  = (body.get("institution") or {}).get("name", "")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
        # Signal the server to shut down
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *args: object) -> None:  # suppress default access logs
        pass


# ─── Public functions ─────────────────────────────────────────────────────────

def connect() -> tuple[str, str]:
    """
    Run the Plaid Link OAuth flow in the user's browser.
    Returns (access_token, institution_name).
    """
    link_token = _create_link_token()

    server = HTTPServer(("localhost", CALLBACK_PORT), _CallbackHandler)
    server.link_token = link_token  # type: ignore[attr-defined]

    url = f"http://localhost:{CALLBACK_PORT}"
    print(f"\nOpening browser → {url}")
    print("Complete the login in your browser, then return here.\n")
    webbrowser.open(url)

    server.serve_forever()   # blocks until callback posts and shuts down

    public_token = _CallbackHandler.public_token
    institution  = _CallbackHandler.institution or "brokerage"

    if not public_token:
        raise RuntimeError("No public token received — connection cancelled.")

    access_token = _exchange_public_token(public_token)
    set_key("plaid-access-token", access_token)
    set_key("plaid-institution",  institution)
    return access_token, institution


def fetch_holdings() -> list[dict]:
    """
    Fetch investment holdings from the connected brokerage via Plaid.
    Returns list of dicts: [{ticker, shares, avg_cost}, ...]
    """
    access_token = get_key("plaid-access-token")
    if not access_token:
        raise RuntimeError(
            "No brokerage connected. Run:  portfolio-tracker connect"
        )

    data = _plaid_post("/investments/holdings/get", {"access_token": access_token})

    # Build security lookup: security_id → ticker
    sec_map: dict[str, str] = {}
    for sec in data.get("securities", []):
        ticker = sec.get("ticker_symbol")
        if ticker and sec.get("type") in ("equity", "etf", "mutual fund"):
            sec_map[sec["security_id"]] = ticker.upper()

    holdings: list[dict] = []
    for h in data.get("holdings", []):
        sec_id   = h.get("security_id", "")
        ticker   = sec_map.get(sec_id)
        quantity = h.get("quantity", 0)
        cost     = h.get("cost_basis")      # total cost basis in $

        if not ticker or not quantity or quantity <= 0:
            continue
        if cost is None or cost <= 0:
            # No cost basis available — use current price
            avg_cost = h.get("institution_price", 0)
        else:
            avg_cost = cost / quantity

        if avg_cost <= 0:
            continue

        holdings.append({
            "ticker":   ticker,
            "shares":   round(quantity, 6),
            "avg_cost": round(avg_cost, 4),
        })

    return holdings
