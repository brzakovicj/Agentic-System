from pathlib import Path

class PromptManager:
    def __init__(self, base_path="prompts"):
        self.base_path = Path(base_path)

    def get(self, name: str, **kwargs):
        file_path = self.base_path / f"{name}.md"
        content = file_path.read_text(encoding="utf-8")
        return content.format(**kwargs)