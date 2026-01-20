import os
import subprocess
import platform
from pathlib import Path

from src.lg2jiuwen_tool.migrator import migrate as migrate_func, MigrationOptions

def migrate(source_path:str,output_dir:str)->str:
    base_dir = os.getenv("BASE_DIR","/")
    source_path = os.path.join(base_dir,source_path.strip(os.path.sep))
    if not source_path.endswith(".py"):
        raise ValueError(f"{source_path} must be a python file")
    if not os.path.exists(source_path):
        raise ValueError(f"{source_path} does not exist")
    output_dir = os.path.join(base_dir,output_dir.strip(os.path.sep))
    os.makedirs(output_dir,exist_ok=True)
    options = MigrationOptions(preserve_comments=True)
    result = migrate_func(source_path,output_dir,options)
    result_file = ""
    for f in result.generated_files:
        if f.endswith(".py"):
            result_file = os.path.relpath(f,base_dir)
            break
    return result_file

def get_file_content(file_path:str)->str:
    base_dir = os.getenv("BASE_DIR","/")
    file_path = os.path.join(base_dir,file_path.strip(os.path.sep))
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} does not exist")
    with open(file_path,"r",encoding="utf-8") as f:
        return f.read()

def run(source_path:str)->str:
    base_dir = os.getenv("BASE_DIR","/")
    source_path = os.path.join(base_dir,source_path.strip(os.path.sep))
    if not source_path.endswith(".py"):
        raise ValueError(f"{source_path} must be a python file")
    if not os.path.exists(source_path):
        raise ValueError(f"{source_path} does not exist")
    cmd = f"source {os.path.join(os.getcwd(),".venv/bin/activate")}"
    if platform.system().lower() == "windows":
        cmd = f"{os.path.join(os.getcwd(),".venv/Scripts/activate.bat")}"
    
    result = subprocess.run(f"{cmd} && python {source_path}",shell=True,cwd=os.path.dirname(source_path),capture_output=True)
    return result.stdout.decode("utf-8")
