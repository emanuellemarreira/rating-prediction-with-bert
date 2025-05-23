import subprocess

scripts = [
    "src.RQ1"
]

print("Executando RQ1...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ1 finalizada.")