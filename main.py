"""- cli entry
- image run
"""
import json
from pathlib import Path

import typer

from vt_action.pipeline import evaluate_image

app = typer.Typer(help="Sight to Action CLI")


@app.command()
def run(image_path: Path):
    """Run the pipeline on an image file."""
    result = evaluate_image(str(image_path))
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
