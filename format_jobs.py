import os
import sqlparse

with open("job_list.txt", "r") as f:
    filenames = [x.strip() for x in f.readlines()]
os.makedirs("job_formatted", exist_ok=True)

for fname in filenames:
    if fname.endswith(".sql"):
        print(fname)
        with open(os.path.join("job", fname), "r") as f:
            x = f.read()
        x = sqlparse.format(x, reindent=True, keyword_case='upper', identifier_case='lower', strip_comments=True, indent_width=4)

        with open(os.path.join("job_formatted", fname), "w") as f:
            f.write(x)
