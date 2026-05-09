from pathlib import Path
from string import Template

BASE_DIR = Path(__file__).resolve().parent

class PromptManager:
    def __init__(self, base_path="prompts"):
        self.base_path = Path(base_path)

    def get(self, name: str, **kwargs):
        file_path = BASE_DIR / f"{name}.md"
        content = file_path.read_text(encoding="utf-8")
        return Template(content).safe_substitute(**kwargs)