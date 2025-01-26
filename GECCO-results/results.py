import os
from collections import defaultdict
import csv

def process_results():
    results = defaultdict(list)
    
    for filename in os.listdir('.'):
        if filename.endswith('.txt'):
            parts = filename[:-4].split(',')
            name = parts[0]
            score = float(parts[2])
            time_part = parts[3].strip()
            try:
                time = float(time_part.split()[0])
                results[name].append((score, time))
            except:
                continue
    
    output_data = []
    for name, values in sorted(results.items()):
        if len(values) > 0:
            scores = [v[0] for v in values]
            times = [v[1] for v in values]
            
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            avg_time = sum(times) / len(times)
            
            output_data.append([
                name,
                f"{best_score:.4f}".replace('.', ','),
                f"{avg_score:.4f}".replace('.', ','),
                f"{avg_time:.1f}".replace('.', ',')
            ])
    
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Instance', 'Best Score', 'Avg Score', 'Avg Time'])
        writer.writerows(output_data)

process_results()