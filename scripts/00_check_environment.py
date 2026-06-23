from importlib import import_module

packages = [
    "numpy", "pandas", "scipy", "sklearn", "statsmodels", "yaml",
    "pyreadstat", "matplotlib", "streamlit", "shap", "pyarrow"
]

missing = []
for package in packages:
    try:
        import_module(package)
    except Exception as exc:  # noqa: BLE001
        missing.append((package, str(exc)))

if missing:
    print("Missing or failed packages:")
    for package, error in missing:
        print(f"- {package}: {error}")
    raise SystemExit(1)

print("Environment check passed.")
