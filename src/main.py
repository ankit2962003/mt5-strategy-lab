"""
Entry point for the project.

Secrets are loaded from environment variables (never hard-coded).
Copy .env.example -> .env and fill in your values.
"""

import os

try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can also be set by the shell.
    pass


def get_secret(name: str) -> str:
    """Read a required secret from the environment, or fail loudly."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def main() -> None:
    # Example: api_key = get_secret("ANGEL_API_KEY")
    print("Project scaffold is working. Replace this with your code.")


if __name__ == "__main__":
    main()
