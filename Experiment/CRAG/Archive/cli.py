# -*- coding: utf-8 -*-
"""
命令行入口：在终端与 CRAG 问答交互
"""
from crag import run_crag


def main():
    print("CRAG 知识库问答（命令行）。输入问题后回车，输入 q 或 exit 退出。\n")
    while True:
        try:
            question = input("您的问题: ").strip()
            if question.lower() in ("q", "exit", "quit"):
                break
            if not question:
                continue
            result = run_crag(question)
            print(f"[动作: {result['action']}] {result['answer']}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}\n")
    print("再见。")


if __name__ == "__main__":
    main()
