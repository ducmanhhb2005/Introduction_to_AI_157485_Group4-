#!/usr/bin/env python3
import os, sys, time

# đưa đúng thư mục src/ (nơi chứa main.py và SearchAlgorithms/) vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# bây giờ có thể import trực tiếp
from main import startGame
from SearchAlgorithms.Bfs import Bfs
from SearchAlgorithms.Dfs import Dfs
from SearchAlgorithms.LocalSearch import local_search
from SearchAlgorithms.Minimax import minimaxAgent
from SearchAlgorithms.Expectimax import ExpectAgent

def benchmark(level, algorithms, map_name, runs):
    stats = {alg: 0 for alg in algorithms}
    for alg in algorithms:
        print(f"\nRunning {alg} on {map_name} ({runs} runs)...")
        wins = 0
        t0 = time.perf_counter()
        for _ in range(runs):
            win, score = startGame(level = level, algorithm=alg, map_name=map_name, is_test = True)
            if win:
                wins += 1
            print(f"{win}, score: {score}")
        elapsed = time.perf_counter() - t0
        rate = wins / runs * 100
        stats[alg] = rate
        print(f"  Win rate: {wins}/{runs} = {rate:.1f}%, time total {elapsed:.2f}s")
    return stats

if __name__ == "__main__":
    level = 1
    algorithms = ["A*", "BFS", "DFS", "Local Search", "Minimax", "Expectimax"]
    map_name   = "../InputMap/Level4/map6.txt"
    runs       = 3

    results = benchmark(level, algorithms, map_name, runs)
    print("\n=== Summary ===")
    for alg, rate in results.items():
        print(f"{alg:12s}: {rate:5.1f}%")