with open("french.txt", encoding="utf-8") as f:
    with open("dialogs_fr.txt", "w", encoding="utf-8") as o:
        o.write("question|answer\n")
        for row in f:
            row = row.strip()
            if "?" in row and (row[-1] != "?" or row.count("?") > 1):
                l = row.strip().split("?")
                o.write(f"{l[0]}?|{l[1].strip()}")
                for i in range(2, len(l)):
                    if len(l[i].strip()) > 0:
                        o.write(f"? {l[i].strip()}")
                o.write("\n")
            else:
                l = row.strip().split(".")
                o.write(f"{l[0]}.|")
                for i in range(1, len(l)):
                    if len(l[i].strip()) > 0:
                        o.write(f"{l[i].strip()} ")
                o.write("\n")



