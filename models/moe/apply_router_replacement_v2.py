
import os
import re

target_file = '/home/conan/workspace/llm_training/models/seqorth_model.py'
new_code_file = '/home/conan/workspace/llm_training/models/new_router_code_v2.py'

# Read original file
with open(target_file, 'r') as f:
    lines = f.readlines()

# Read new code
with open(new_code_file, 'r') as f:
    new_code = f.read()

# Identify the range to replace
# We look for "class SeqorthRouter(nn.Module):" and "iterations = 0"
# Based on previous turn and file operations.

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "class SeqorthRouter(nn.Module):" in line:
        start_idx = i
    if "iterations = 0" in line:
        end_idx = i
        # Check if we found both and if end_idx is after start_idx
        if start_idx != -1 and end_idx > start_idx:
             break

if start_idx == -1 or end_idx == -1:
    print(f"Error: Could not find start ({start_idx}) or end ({end_idx}) markers.")
    exit(1)

print(f"Replacing lines {start_idx+1} to {end_idx} (non-inclusive of end_idx in theory, but we splice lists)")

# Construct new content
new_lines = lines[:start_idx]
new_lines.append(new_code + "\n\n")
new_lines.extend(lines[end_idx:])

# Write back
with open(target_file, 'w') as f:
    f.writelines(new_lines)

print("Successfully replaced SeqorthRouter class with new simplified version.")
