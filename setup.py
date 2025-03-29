from setuptools import setup, find_packages

setup(
    name="organizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "pydantic>=2.6.1",
        "python-dotenv>=1.0.0",
        "google-genai>=1.8.0",
        "rich>=13.7.0",
        "llama-cpp-python>=0.2.56",
    ],
    entry_points={
        "console_scripts": [
            "organizer=organizer.cli:main",
        ],
    },
)