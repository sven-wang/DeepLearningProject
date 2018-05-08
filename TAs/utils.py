import os

files = os.listdir(".")

persons = []
for each in files:
    if each.endswith(".npy"):
        persons.append(each)

tuples = []

for i in range(len(persons)):
    file1 = persons[i]
    name1 = file1[:-5]
    for j in range(i + 1, len(persons)):
        file2 = persons[j]
        name2 = file2[:-5]
        if name1 == name2:
            tuples.append((file1, file2, "target"))
        else:
            tuples.append((file1, file2, "non-target"))

with open("tas.txt", "w") as f:
    for each in tuples:
        f.write(each[0] + " " + each[1] + " " + each[2] + "\n")

