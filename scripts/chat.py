#!/usr/bin/env python3
"""
Interactive chat client for SGLang server.

A simple CLI tool to chat with your deployed LLM and test inference speed.
Uses the OpenAI-compatible API provided by SGLang.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install dependencies with: uv sync")
    sys.exit(1)


class Colors:
    """ANSI color codes."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class SGLangChat:
    """Simple chat client for SGLang server."""

    def __init__(self, host: str, port: int, model: str):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.conversation_history = []

    def health_check(self) -> bool:
        """Check if server is responding."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"{Colors.RED}Error connecting to server: {e}{Colors.RESET}")
            return False

    def get_models(self) -> list:
        """Get available models from server."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json().get("data", [])
        except Exception:
            pass
        return []

    def chat(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Optional[str]:
        """Send a chat message and get response."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Prepare request
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=120)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]

                # Add assistant response to history
                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_message}
                )

                # Calculate metrics
                elapsed = end_time - start_time
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Print metrics
                print(f"\n{Colors.YELLOW}[Metrics]{Colors.RESET}")
                print(f"  Time: {elapsed:.2f}s")
                if completion_tokens > 0:
                    tokens_per_sec = completion_tokens / elapsed
                    print(f"  Speed: {tokens_per_sec:.1f} tokens/s")
                if total_tokens > 0:
                    print(f"  Tokens: {total_tokens} total")

                return assistant_message
            else:
                print(
                    f"{Colors.RED}Error: Server returned {response.status_code}{Colors.RESET}"
                )
                print(response.text)
                return None

        except requests.exceptions.Timeout:
            print(f"{Colors.RED}Error: Request timed out{Colors.RESET}")
            return None
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            return None

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print(f"{Colors.GREEN}Conversation history cleared{Colors.RESET}")


def print_banner(host: str, port: int, model: str):
    """Print welcome banner."""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}SGLang Interactive Chat{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"\nServer: {Colors.BLUE}{host}:{port}{Colors.RESET}")
    print(f"Model:  {Colors.BLUE}{model}{Colors.RESET}")
    print(f"\n{Colors.YELLOW}Commands:{Colors.RESET}")
    print("  /reset  - Clear conversation history")
    print("  /quit   - Exit chat")
    print("  /help   - Show this help message")
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def main():
    """Run interactive chat client."""
    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Parse arguments
    parser = argparse.ArgumentParser(description="Interactive chat with SGLang server")
    parser.add_argument(
        "--host",
        default=os.getenv("SERVER_HOST", "localhost"),
        help="Server host (default: localhost or SERVER_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SERVER_PORT", "30000")),
        help="Server port (default: 30000 or SERVER_PORT env var)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3-70B-Instruct"),
        help="Model name (default: MODEL_PATH env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    args = parser.parse_args()

    # Create chat client
    chat_client = SGLangChat(args.host, args.port, args.model)

    # Health check
    print(f"Checking server health at {args.host}:{args.port}...")
    if not chat_client.health_check():
        print(f"\n{Colors.RED}Server is not responding!{Colors.RESET}")
        print(f"\nMake sure the SGLang server is running:")
        print(f"  make start-server")
        print(f"\nOr check the server address: {args.host}:{args.port}")
        sys.exit(1)

    print(f"{Colors.GREEN}✓ Server is online{Colors.RESET}")

    # Check available models
    models = chat_client.get_models()
    if models:
        print(f"\nAvailable models:")
        for model in models:
            model_id = model.get("id", "unknown")
            print(f"  - {model_id}")

    # Print banner
    print_banner(args.host, args.port, args.model)

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}You: {Colors.RESET}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print(f"\n{Colors.BLUE}Goodbye!{Colors.RESET}\n")
                break
            elif user_input.lower() == "/reset":
                chat_client.reset_conversation()
                continue
            elif user_input.lower() == "/help":
                print_banner(args.host, args.port, args.model)
                continue

            # Send message
            print(f"\n{Colors.BLUE}Assistant: {Colors.RESET}", end="", flush=True)
            response = chat_client.chat(
                user_input, max_tokens=args.max_tokens, temperature=args.temperature
            )

            if response:
                print(response)
            else:
                print(f"{Colors.RED}Failed to get response{Colors.RESET}")

            print()  # Extra newline for readability

        except KeyboardInterrupt:
            print(f"\n\n{Colors.BLUE}Goodbye!{Colors.RESET}\n")
            break
        except EOFError:
            print(f"\n\n{Colors.BLUE}Goodbye!{Colors.RESET}\n")
            break


if __name__ == "__main__":
    main()
