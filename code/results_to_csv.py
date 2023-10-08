import csv
import re

header = ["name", "ET", "TC", "WT"]

csv_file = open("../test_results.csv", "w", encoding="UTF8")
writer = csv.writer(csv_file)
writer.writerow(header)

with open("../save_file.txt","r") as f:
    text = f.read().splitlines()

for row in text[1:]:
    row_split = row.split("-")
    name = row_split[0]
    results = row_split[1]
    res = [name]
    for el in results.split(","):
        numbers = re.findall("\d+\.\d+", el)
        res.append(str(round(100*float(numbers[0]), 2)) + " ± " + str(round(100*float(numbers[1]), 2)))
    writer.writerow(res)

csv_file.close()
