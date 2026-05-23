import json
import httpx
import time
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    console.print(f"\n[bold blue]Evaluating Dataset: {file_path}[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Message snippet", width=40)
    table.add_column("True Label")
    table.add_column("Prediction")
    table.add_column("Confidence")
    table.add_column("Latency")
    table.add_column("Correct?", justify="center")

    correct = 0
    total = len(dataset)

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        # Quick health check
        try:
            client.get("/docs")
        except httpx.ConnectError:
            console.print("[bold red]Error: FastAPI server is not running on http://127.0.0.1:8000[/bold red]")
            console.print("Please run [italic]uvicorn modules.app:app[/italic] first.")
            return

        for item in dataset:
            text = item["text"]
            true_label = item["true_label"]
            
            resp = client.post("/predict", json={"text": text})
            if resp.status_code == 200:
                data = resp.json()
                pred = data["label"]
                conf = data["confidence"]
                lat = data["latency_ms"]
                
                is_correct = (pred == true_label)
                if is_correct:
                    correct += 1
                    status_icon = "[green]YES[/green]"
                else:
                    status_icon = "[red]NO[/red]"

                snippet = text[:37] + "..." if len(text) > 40 else text
                
                # Highlight prediction in red if it was wrong
                pred_display = f"[green]{pred}[/green]" if is_correct else f"[red]{pred}[/red]"
                
                table.add_row(
                    snippet,
                    true_label,
                    pred_display,
                    f"{conf:.2f}",
                    f"{lat:.1f}ms",
                    status_icon
                )
            else:
                console.print(f"[red]API Error {resp.status_code} for text: {text}[/red]")

    console.print(table)
    accuracy = (correct / total) * 100
    color = "green" if accuracy == 100 else "yellow"
    console.print(f"Accuracy on this dataset: [bold {color}]{correct}/{total} ({accuracy:.1f}%)[/bold {color}]\n")

if __name__ == "__main__":
    evaluate_dataset("test_datasets/modern_tricky.json")
