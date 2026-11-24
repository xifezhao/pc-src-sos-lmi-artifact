# Troubleshooting

- **`ModuleNotFoundError: No module named 'pcsrc'`**: Run `pip install -e .` or prefix commands with `PYTHONPATH=src`.
- **Clarabel wheel build issues on older pip**: Upgrade pip first (`python3 -m pip install --upgrade pip`) before `pip install -r requirements-lock.txt`.
- **Headless Matplotlib display errors**: Set `MPLBACKEND=Agg`.
