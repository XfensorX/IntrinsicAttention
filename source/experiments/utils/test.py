import json
from collections import Counter


def analyze_graph(file_path):
    """
    Parse PyTorch graph JSON file and extract operation counts.

    Args:
        file_path: Path to the JSON file containing the graph

    Returns:
        Counter object with operation type counts
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Count operation types
    op_counts = Counter()
    for node in data.get("nodes", []):
        if "op" in node:
            op_counts[node["op"]] += 1

    return op_counts


# Analyze both files
iter1_ops = analyze_graph("iter1_no_outer.json")
iter2_ops = analyze_graph("iter2_no_outer.json")

# Find operations more frequent in iter2
more_frequent_in_iter2 = {}
for op, count2 in iter2_ops.items():
    count1 = iter1_ops.get(op, 0)
    if count2 > count1:
        more_frequent_in_iter2[op] = count2 - count1

# Sort by difference in frequency
sorted_ops = sorted(more_frequent_in_iter2.items(), key=lambda x: x[1], reverse=True)

# Print results
print("\nOperationen, die in iter2.json häufiger vorkommen als in iter1.json:")
print("=" * 70)
print(f"{'Operation':<40} | {'Differenz':<10}")
print("-" * 70)
for op, diff in sorted_ops:
    print(f"{op:<40} | +{diff:<10}")

# Print summary
total_diff = sum(more_frequent_in_iter2.values())
print("\nZusammenfassung:")
print(
    f"Insgesamt {len(more_frequent_in_iter2)} Operationstypen kommen in iter2.json häufiger vor"
)
print(f"Gesamtdifferenz: +{total_diff} zusätzliche Operationen in iter2.json")
