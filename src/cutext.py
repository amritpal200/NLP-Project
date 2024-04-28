import subprocess
import os
import json
from typing import Union

ROOT_DIR = os.path.dirname(os.path.abspath(""))

def run_command(
		command: str,
		verbosity: int=0
) -> None:
	try:
		result = subprocess.run(command, shell=True, text=True, capture_output=True)
		if verbosity > 0: print("Output:", result.stdout)
		if verbosity > 1:
			print("Error:", result.stderr)
			print("Return code:", result.returncode)
	except Exception as e:
		if verbosity > 0: print("An error occurred:", e)

def get_cutext_path() -> Union[str, None]:
	with open(os.path.join(ROOT_DIR, "config", "config.json"), "r") as f:
		config = json.load(f)
	return config.get("cutext_path")

def run_cutext(
		input_path: str,
		output_dir: str,
		verbosity: int=0,
		cutext_path: str=None
) -> None:
	if not cutext_path:
		cutext_path = get_cutext_path()
	if not cutext_path:
		raise Exception("cutext path is not defined in config.json")
	elif not os.path.exists(cutext_path):
		raise Exception("cutext path does not exist")
	main_dir = str(os.path.join(cutext_path, 'cutext', 'main'))
	if not os.path.exists(input_path):
		raise Exception("Input file does not exist")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# check OS
	if os.name == 'nt': # Windows
		if not output_dir.endswith("\\"):
			output_dir += "\\"
	elif os.name == 'posix': # Linux or Mac
		if not output_dir.endswith("/"):
			output_dir += "/"
	command = f"java -jar cutext.jar -TM -generateTextFile true -inputFile {input_path} -routeTextFileHashTerms {output_dir}"
	run_command("cd "+main_dir+" && "+command, verbosity)

def process_input(
		cutext_path: str=None
) -> None:
	input_path = os.path.join(ROOT_DIR, "temp", "cutext_in.txt")
	output_dir = os.path.join(ROOT_DIR, "temp")
	run_cutext(input_path, output_dir, cutext_path=cutext_path)

def clean_temp(
		all: bool=False
) -> None:
	temp_dir = os.path.join(ROOT_DIR, "temp")
	if not os.path.exists(temp_dir): return
	for file in os.listdir(temp_dir):
		if file.startswith(".fuse_hidden") or all:
			os.remove(os.path.join(temp_dir, file))
