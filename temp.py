import random
import pickle

# Write all triplets to a file,
# input_file = open("triplets.csv", "r")
# triplets = input_file.readlines()
# input_file.close()
#
# samples = random.choices(triplets, k=int(0.1 * len(triplets)))
# output_file = open("triplets_sample.csv", 'w')
# for triplet in samples:
#     output_file.write(triplet)
# output_file.close()

input_file = open("enrol.txt", "r")
persons = input_file.readlines()
input_file.close()

map = {}
for person in persons:
    file_name = person.strip().split()[-1]
    person_id = file_name[0:5]
    if person_id in map:
        print(person_id)
        print("Duplicates!")
    map[person_id] = file_name

with open("enrol_file_map.pickle", 'wb') as handle:
    pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)