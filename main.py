#!/usr/bin/env python3
import os
import sys
import io
import argparse
from dotenv import load_dotenv
from chains.rag_chain import RAGChain
from rich.console import Console
from rich.panel import Panel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Knowledge System")
    parser.add_argument("--query", "-q", default="公司年假政策是什么?", help="查询问题")
    parser.add_argument("--docs", "-d", default="sample_docs", help="文档目录")
    parser.add_argument("--model", "-m", default="gpt-4", help="使用的模型")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        f"[bold cyan]企业知识库RAG系统[/bold cyan]\n"
        f"查询: {args.query}\n"
        f"文档目录: {args.docs}",
        border_style="cyan"
    ))
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold yellow]Warning: 未设置 API_KEY，使用模拟模式[/bold yellow]")
    
    chain = RAGChain(model=args.model)
    result = chain.run(args.query, args.docs)
    
    answer = chain.get_final_answer()
    
    console.print("\n" + "="*50)
    console.print("[bold]最终答案:[/bold]\n")
    console.print(answer)


if __name__ == "__main__":
    main()