import subprocess

scripts = [
    "src.RQ2"
]

print("Executando RQ2...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ2 finalizada.")