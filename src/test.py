#!/usr/bin/env python3
import os, sys, time

# đưa đúng thư mục src/ (nơi chứa main.py và SearchAlgorithms/) vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# bây giờ có thể import trực tiếp
from main import startGame

def benchmark(level, algorithms, map_name, runs):
    stats = {}
    for alg in algorithms:
        print(f"\nRunning {alg} on {map_name} ({runs} runs)...")
        wins = 0
        t0 = time.perf_counter()
        total_score = 0

        for _ in range(runs):
            win, score = startGame(level=level, algorithm=alg, map_name=map_name, is_test=True, is_render=False)
            total_score += score
            if win:
                wins += 1
            print(f"{win}, score: {score}")

        elapsed = time.perf_counter() - t0
        rate = wins / runs * 100
        avg_score = total_score / runs
        avg_time = elapsed / runs

        # Lưu stats dưới dạng dictionary
        stats[alg] = {
            'win_rate': rate,
            'avg_score': avg_score,
            'avg_time': avg_time,
            'wins': wins,
            'total_runs': runs
        }

        print(f"  win rate: {wins}/{runs} = {rate:.1f}%, average score: {avg_score:.2f}, average time: {avg_time:.4f}s")
    
    return stats

if __name__ == "__main__":
    level = 1
    algorithms = ["BFS", "DFS", "Local Search"]
    # algorithms = ["BFS", "DFS", "A*"]
    map_name = "../InputMap/Level4/map6.txt"
    runs = 200

    results = benchmark(level, algorithms, map_name, runs)
    
    print("\n" + "="*80)
    print("                              SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<15} {'Win Rate':<10} {'Avg Score':<12} {'Avg Time (s)':<15}")
    print("-" * 80)
    
    for alg, stats in results.items():
        print(f"{alg:<15} {stats['win_rate']:>8.1f}% {stats['avg_score']:>10.2f} {stats['avg_time']:>13.4f}")
    
    print("="*80)
    
    # Tìm thuật toán tốt nhất theo từng tiêu chí
    best_win_rate = max(results.items(), key=lambda x: x[1]['win_rate'])
    best_avg_score = max(results.items(), key=lambda x: x[1]['avg_score'])
    fastest = min(results.items(), key=lambda x: x[1]['avg_time'])
    
    print(f"\nBest Win Rate:  {best_win_rate[0]} ({best_win_rate[1]['win_rate']:.1f}%)")
    print(f"Best Avg Score: {best_avg_score[0]} ({best_avg_score[1]['avg_score']:.2f})")
    print(f"Fastest:        {fastest[0]} ({fastest[1]['avg_time']:.4f}s)")