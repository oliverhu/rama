import sys
import re
for line in sys.stdin:
    line = line[0:-1]
    m = re.search('.*data: (.*)', line)
    if m:
        sys.stdout.write(m.group(1).replace('\\n', '\n'))
        sys.stdout.flush()