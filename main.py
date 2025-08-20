import argparse
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import get_settings
from app.chatbot import StudyAssistant


console = Console()


def print_structured(response) -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left")
    table.add_row(f"[bold]Answer[/bold]\n{response.answer}")

    if response.key_points:
        bullets = "\n".join([f"- {p}" for p in response.key_points])
        table.add_row(f"[bold]Key Points[/bold]\n{bullets}")

    if response.suggested_questions:
        sq = "\n".join([f"- {q}" for q in response.suggested_questions])
        table.add_row(f"[bold]Suggested Questions[/bold]\n{sq}")

    if response.references:
        refs = "\n".join([f"- {r}" for r in response.references])
        table.add_row(f"[bold]References[/bold]\n{refs}")

    console.print(Panel.fit(table, title="Study Assistant"))


def build_assistant(model: str, embed_model: str, persist_path: str, session_id: str) -> StudyAssistant:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Create .env or set the environment variable.")
    assistant = StudyAssistant(
        groq_api_key=settings.groq_api_key,
        model=model,
        embeddings_model=embed_model,
        persist_path=persist_path,
    )
    return assistant


def interactive_chat(assistant: StudyAssistant, session_id: str) -> None:
    console.print("[bold cyan]Type your question. Ctrl+C to exit.[/bold cyan]")
    while True:
        try:
            user = console.input("[bold green]You> [/bold green]")
            if not user.strip():
                continue
            response = assistant.ask(user, session_id=session_id)
            print_structured(response)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Study Assistant (LangChain + Groq + HF)")
    parser.add_argument("--once", type=str, help="Run a single prompt and exit", default=None)
    parser.add_argument("--model", type=str, default="llama3-70b-8192")
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--persist-path", type=str, default="./storage/memory_db")
    parser.add_argument("--session-id", type=str, default="default")
    args = parser.parse_args()

    os.makedirs(args.persist_path, exist_ok=True)
    assistant = build_assistant(
        model=args.model,
        embed_model=args.embed_model,
        persist_path=args.persist_path,
        session_id=args.session_id,
    )

    if args.once:
        response = assistant.ask(args.once, session_id=args.session_id)
        print_structured(response)
    else:
        interactive_chat(assistant, session_id=args.session_id)


if __name__ == "__main__":
    main()
