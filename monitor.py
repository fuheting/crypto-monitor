import urllib.request
import json
import time
import csv
from datetime import datetime
from rich.console import Console
from rich.table import Table

URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
CSV_FILE = "btc_data.csv"

console = Console()


def fetch_btc_price():
    with urllib.request.urlopen(URL) as response:
        data = json.loads(response.read().decode())
    return float(data["price"])


def append_to_csv(timestamp, price):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, price])


def get_table(price):
    table = Table(title="BTC/USD Price Monitor")
    table.add_column("Time", style="cyan")
    table.add_column("Price (USD)", style="green", justify="right")
    table.add_row(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"${price:,.2f}")
    return table


if __name__ == "__main__":
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price"])

    while True:
        try:
            price = fetch_btc_price()
            timestamp = datetime.now().isoformat()
            append_to_csv(timestamp, price)
            console.clear()
            console.print(get_table(price))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        time.sleep(60)
