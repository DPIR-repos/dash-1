input_file = "conda_requirements.txt"
output_file = "requirements.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split("=")
        if len(parts) >= 2:
            name = parts[0]
            version = parts[1]
            pip_line = f"{name}=={version}\n"
            outfile.write(pip_line)

print(f"Archivo '{output_file}' creado.")
