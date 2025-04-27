# Written by ChatGPT

import re

# Suppose your text is stored in a file
filename = "log"  # replace with your actual filename

with open(filename, "r") as f:
    content = f.read()

# Use regex to find timeElapsed
match = re.search(r'timeElapsed:\s*(\d+)\s*ms', content)

if match:
    time_elapsed = int(match.group(1))
    print(time_elapsed)