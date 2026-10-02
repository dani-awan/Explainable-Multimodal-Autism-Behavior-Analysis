"""Notebook-derived standalone stage. Source cells: [79]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    from datetime import datetime, timezone
    import json
    import subprocess
    import sys


    DOCS = PROJECT / "docs"
    DOCS.mkdir(parents=True, exist_ok=True)

    import os
    POSE_PYTHON = PROJECT / (".venv-pose/Scripts/python.exe" if os.name == "nt" else ".venv-pose/bin/python")

    # Collect installed distribution versions without importing heavy libraries.
    environment_code = r'''
    import importlib.metadata as metadata
    import json
    import platform

    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version

    print(json.dumps({
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }))
    '''

    environments = {}
    python_paths = {
        "main": Path(sys.executable),
        "pose": POSE_PYTHON,
    }

    for name, executable in python_paths.items():
        if not executable.is_file():
            raise FileNotFoundError(f"{name} Python executable not found.")

        result = subprocess.run(
            [str(executable), "-c", environment_code],
            capture_output=True,
            text=True,
            check=True,
        )
        environment = json.loads(result.stdout)
        environments[name] = environment

        # Save version snapshots without personal installation paths.
        requirements = "\n".join(
            f"{package}=={version}"
            for package, version in environment["packages"].items()
        ) + "\n"

        path = PROJECT / f"requirements-{name}-snapshot.txt"
        path.write_text(requirements, encoding="utf-8")

        print(
            f"{name}: Python {environment['python_version']} | "
            f"{len(environment['packages'])} installed distributions"
        )
        print(f"Saved: {path}")

    # Inventory project scripts and check syntax without executing them.
    inventory = []

    for folder_name in ("scripts", "dashboard"):
        folder = PROJECT / folder_name
        if not folder.is_dir():
            continue

        for path in sorted(folder.rglob("*.py")):
            if any(
                part.startswith(".venv") or part == "__pycache__"
                for part in path.relative_to(PROJECT).parts
            ):
                continue

            status = "PASS"
            error = ""
            try:
                source = path.read_text(encoding="utf-8-sig")
                compile(source, str(path), "exec")
            except Exception as exception:
                status = "FAIL"
                error = f"{type(exception).__name__}: {exception}"

            inventory.append({
                "file": path.relative_to(PROJECT).as_posix(),
                "syntax_status": status,
                "error": error,
            })

    project_notebooks = [
        path.relative_to(PROJECT).as_posix()
        for path in sorted(PROJECT.rglob("*.ipynb"))
        if not any(
            part.startswith(".venv") or part == ".ipynb_checkpoints"
            for part in path.relative_to(PROJECT).parts
        )
    ]

    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "environments": environments,
        "script_inventory": inventory,
        "project_notebooks": project_notebooks,
        "fresh_environment_installation_verified": False,
        "full_pipeline_reproduction_verified": False,
    }

    record_path = DOCS / "reproducibility_inventory.json"
    record_path.write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    notes = """# Reproducibility Status

    ## Environment snapshots

    - requirements-main-snapshot.txt: installed versions from the notebook environment.
    - requirements-pose-snapshot.txt: installed versions from the pose environment.
    - docs/reproducibility_inventory.json: Python versions and script inventory.

    These files record installed versions. They are not dependency lockfiles.
    A fresh installation has not yet been verified. The main environment may
    contain packages unrelated to this project.

    ## Runtime separation

    - Main environment: audio features, handcrafted visual features,
      model evaluation, explainability, and Streamlit.
    - Pose environment: MediaPipe pose and visual embedding scripts.
    - FFmpeg and FFprobe are external dependencies and must be installed separately.

    ## Remaining verification

    - Identify processing steps that still exist only in notebook cells.
    - Convert those steps into standalone scripts with project-relative paths.
    - Preserve saved fold assignments, feature order, labels, and thresholds.
    - Verify installation in a clean environment.
    - Verify pipeline execution before claiming full reproducibility.

    Python syntax checks do not verify runtime behavior or scientific results.
    """

    notes_path = DOCS / "REPRODUCIBILITY_STATUS.md"
    notes_path.write_text(notes, encoding="utf-8")

    print("\nProject script inventory:")
    for item in inventory:
        print(f"{item['syntax_status']}: {item['file']}")
        if item["error"]:
            print(f"  {item['error']}")

    print("\nNotebooks inside the project:")
    for notebook in project_notebooks:
        print(notebook)
    if not project_notebooks:
        print("None found; the working notebook may be outside the project.")

    print(f"\nSaved: {record_path}")
    print(f"Saved: {notes_path}")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=[],
        produces=['requirements-main-snapshot.txt', 'requirements-pose-snapshot.txt', 'docs/reproducibility_inventory.json', 'docs/REPRODUCIBILITY_STATUS.md'],
        external=(),
        originals=False,
    )
