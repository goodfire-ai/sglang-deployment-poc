#!/usr/bin/env python3
"""
Validate environment configuration for SGLang deployment.

This script checks that all required environment variables are set
and optionally validates the HuggingFace token.
"""

import os
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_success(message):
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def print_error(message):
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def print_info(message):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")


def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        print_error(".env file not found")
        if env_example.exists():
            print_info(f"Copy {env_example} to .env and fill in your values:")
            print(f"  cp {env_example} .env")
        return False

    print_success(".env file exists")
    return True


def check_required_vars():
    """Check required environment variables."""
    required_vars = {
        "HF_TOKEN": "HuggingFace token (get from https://huggingface.co/settings/tokens)",
    }

    optional_vars = {
        "MODEL_PATH": "Model path (defaults to meta-llama/Meta-Llama-3-70B-Instruct)",
        "SERVER_HOST": "Server host (defaults to 0.0.0.0)",
        "SERVER_PORT": "Server port (defaults to 30000)",
        "TENSOR_PARALLEL_SIZE": "Tensor parallelism size (defaults to 4)",
    }

    print(f"\n{Colors.BOLD}Required Variables:{Colors.RESET}")
    all_required_set = True

    for var, description in required_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            # Mask sensitive tokens
            display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            print_success(f"{var}={display_value} ({description})")
        else:
            print_error(f"{var} not set - {description}")
            all_required_set = False

    print(f"\n{Colors.BOLD}Optional Variables:{Colors.RESET}")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print_success(f"{var}={value} ({description})")
        else:
            print_warning(f"{var} not set - {description}")

    return all_required_set


def validate_hf_token():
    """Validate HuggingFace token by attempting to authenticate."""
    token = os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user_info = api.whoami()
        print_success(f"HuggingFace token valid (user: {user_info['name']})")
        return True
    except ImportError:
        print_warning(
            "huggingface-hub not installed, skipping token validation"
        )
        print_info("Install dependencies to validate: uv sync")
        return True
    except Exception as e:
        print_error(f"HuggingFace token validation failed: {e}")
        print_info("Check your token at: https://huggingface.co/settings/tokens")
        return False


def validate_model_access():
    """Check if we can access the specified model."""
    model_path = os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3-70B-Instruct")
    token = os.getenv("HF_TOKEN")

    # Skip validation for local paths
    if not model_path.startswith(("meta-llama/", "mistralai/", "huggingface/")):
        if Path(model_path).exists():
            print_success(f"Local model path exists: {model_path}")
        else:
            print_warning(f"Local model path does not exist: {model_path}")
        return True

    if not token or token == "your_huggingface_token_here":
        print_warning(f"Cannot validate model access without HF_TOKEN: {model_path}")
        return True

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        model_info = api.model_info(model_path)
        print_success(f"Model accessible: {model_path}")

        if model_info.gated:
            print_info(f"Model is gated - ensure you have accepted terms at:")
            print(f"  https://huggingface.co/{model_path}")

        return True
    except ImportError:
        print_warning("huggingface-hub not installed, skipping model validation")
        return True
    except Exception as e:
        print_warning(f"Could not validate model access: {e}")
        print_info(f"Check model at: https://huggingface.co/{model_path}")
        return True


def main():
    """Run all validation checks."""
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}SGLang Environment Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print_info("Loaded environment from .env file")
    except ImportError:
        print_warning("python-dotenv not installed, using system environment only")
        print_info("Install with: uv pip install python-dotenv")

    print()

    # Run checks
    checks = [
        ("Environment file", check_env_file),
        ("Required variables", check_required_vars),
        ("HuggingFace token", validate_hf_token),
        ("Model access", validate_model_access),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            print()
        except Exception as e:
            print_error(f"Check failed with error: {e}")
            results.append((name, False))
            print()

    # Summary
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Summary:{Colors.RESET}\n")

    all_passed = all(result for _, result in results if result is not None)

    for name, result in results:
        if result is True:
            print_success(f"{name}: OK")
        elif result is False:
            print_error(f"{name}: FAILED")
        else:
            print_warning(f"{name}: SKIPPED")

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.RESET}")
        print(f"You're ready to deploy SGLang.\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some checks failed{Colors.RESET}")
        print(f"Fix the errors above before deploying.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
