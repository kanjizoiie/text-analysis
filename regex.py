import re
pattern = "\w+"
prog = re.compile(pattern)
print(prog.match("AAAAAA"))