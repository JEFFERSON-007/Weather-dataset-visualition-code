import os
import subprocess
import sys
import re

APP_FILE = "app.py"   # Change if your file name is different
REQUIREMENTS_FILE = "requirements.txt"


# -------------------------------------------
# 1. Extract imports from the project folder
# -------------------------------------------
def extract_imports(pyfile):
    with open(pyfile, "r", encoding="utf-8") as f:
        code = f.read()

    imports = set()
    matches = re.findall(r"^\s*(?:import|from)\s+([\w\-\.]+)", code, re.MULTILINE)

    for m in matches:
        pkg = m.split(".")[0]  # keep root module only
        imports.add(pkg)

    return imports


# -------------------------------------------
# 2. Convert imports → pip packages
# -------------------------------------------
def map_to_pip(imports):
    # Most common mappings
    known_map = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pandas": "pandas",
        "numpy": "numpy",
        "multipart": "python-multipart",
        "starlette": "starlette",
        "pydantic": "pydantic",
        "sqlalchemy": "sqlalchemy",
    }

    pip_pkgs = set()

    for i in imports:
        pip_pkgs.add(known_map.get(i, i))  # default: install same name

    return pip_pkgs


# -------------------------------------------
# 3. Write requirements.txt
# -------------------------------------------
def write_requirements(pkgs):
    with open(REQUIREMENTS_FILE, "w") as f:
        for p in pkgs:
            f.write(p + "\n")
    print(f"Generated {REQUIREMENTS_FILE}")


# -------------------------------------------
# 4. Install the requirements
# -------------------------------------------
def install_requirements():
    print("\nInstalling requirements...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
    subprocess.run(cmd)
    print("All requirements installed.\n")


# -------------------------------------------
# 5. Run the FastAPI app
# -------------------------------------------
def run_app():
    print("Starting FastAPI server...\n")
    cmd = [sys.executable, "-m", "uvicorn", f"{APP_FILE.replace('.py','')}:app", "--reload"]
    subprocess.run(cmd)


# -------------------------------------------
# MAIN
# -------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(APP_FILE):
        print(f"ERROR: Could not find {APP_FILE}")
        exit()

    print("Extracting imports...")
    imports = extract_imports(APP_FILE)
    pip_pkgs = map_to_pip(imports)

    write_requirements(pip_pkgs)
    install_requirements()
    run_app()
