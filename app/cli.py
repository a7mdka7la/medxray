"""Terminal demo for both modes.

Usage:
    python -m app.cli report --image path/to/cxr.jpg --strategy medgemma_direct
    python -m app.cli qa --image path/to/cxr.jpg --question "Is there pleural effusion?"
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from src.pipeline.rag_qa import RAGQAPipeline
from src.pipeline.report_generation import ReportGenerator

app = typer.Typer(help="MedXray CLI — two modes: report / qa")
console = Console()


@app.command()
def report(
    image: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Chest X-ray path"),
    strategy: str = typer.Option("medgemma_direct", help="medgemma_direct | colpali_rag | clip_rag"),
    indication: str = typer.Option("", help="Optional clinical indication"),
    k: int = typer.Option(3, help="Top-k retrieved reports (RAG only)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of pretty text"),
) -> None:
    gen = ReportGenerator()
    result = gen.generate(image, strategy=strategy, indication=indication or None, k=k)

    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return

    console.print(Panel.fit(result.raw_text, title=f"Report ({strategy})  ·  {result.latency_s:.2f}s"))
    if result.retrieved:
        console.print("\n[bold]Retrieved references:[/bold]")
        for i, h in enumerate(result.retrieved, 1):
            console.print(f"  {i}. {h.image_id}  ·  score={h.score:.3f}")


@app.command()
def qa(
    image: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    question: str = typer.Option(..., help="Clinical question"),
    backend: str = typer.Option("colpali", help="colpali | clip"),
    k: int = typer.Option(4, help="Top-k retrieved reports"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    pipe = RAGQAPipeline(backend=backend)
    result = pipe.answer(image, question, k=k)

    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return

    console.print(Panel.fit(result.answer, title=f"Answer  ·  backend={backend}  ·  {result.latency_s:.2f}s"))
    if result.retrieved:
        console.print("\n[bold]Retrieved references:[/bold]")
        for i, h in enumerate(result.retrieved, 1):
            console.print(f"  {i}. {h.image_id}  ·  score={h.score:.3f}")


if __name__ == "__main__":
    app()
