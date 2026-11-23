from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run_step(script: str):
    path = ROOT / script

    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    print(
        f"\n{'=' * 72}\n"
        f"Running {script}\n"
        f"{'=' * 72}",
        flush=True,
    )

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )


def main():
    run_step("src/prepare_data.py")
    run_step("src/analysis.py")
    run_step("src/plots.py")

    print("\nPipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
