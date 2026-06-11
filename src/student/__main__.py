"""Entry point: python -m student <command>"""
import fire
from student.cli import StudentCLI

if __name__ == "__main__":
    fire.Fire(StudentCLI)