import random
import pickle

# Write all triplets to a file,
input_file = open("triplets_misclassified.csv", "r")
triplets = input_file.readlines()
input_file.close()

samples = random.choices(triplets, k=10000)

input_file = open("triplets_general.csv", "r")
triplets = input_file.readlines()
input_file.close()

samples.extend(random.choices(triplets, k=50000))

output_file = open("triplets.csv", 'w')
for triplet in samples:
    output_file.write(triplet)
output_file.close()

# input_file = open("enrol.txt", "r")
# persons = input_file.readlines()
# input_file.close()
#
# map = {}
# for person in persons:
#     file_name = person.strip().split()[-1]
#     person_id = file_name[0:5]
#     if person_id in map:
#         print(person_id)
#         print("Duplicates!")
#     map[person_id] = file_name
#
# with open("enrol_file_map.pickle", 'wb') as handle:
#     pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# input_file = open("test.txt", "r")
# persons = input_file.readlines()
# input_file.close()
#
# map = set([])
# for person in persons:
#     file_name = person.strip().split()[-1]
#     id = file_name[0:7]
#     if id in map:
#         print(id)
#         print("Duplicates!")
#     map.add(id)
#
# with open("test_file_set.pickle", 'wb') as handle:
#     pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)

# input_file = open("enrol.txt", "r")
# persons = input_file.readlines()
# input_file.close()
#
# map = {}
# for person in persons:
#     prob = random.random()
#     if prob > 0.025:
#         continue
#     file_name = person.strip().split()[-1]
#     person_id = file_name[0:5]
#     if person_id in map:
#         print(person_id)
#         print("Duplicates!")
#     map[person_id] = file_name
#
# with open("enrol_file_map_sample.pickle", 'wb') as handle:
#     pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# input_file = open("test.txt", "r")
# persons = input_file.readlines()
# input_file.close()
#
# map = set([])
# for person in persons:
#     prob = random.random()
#     if prob > 0.25:
#         continue
#     file_name = person.strip().split()[-1]
#     id = file_name[0:7]
#     if id in map:
#         print(id)
#         print("Duplicates!")
#     map.add(id)
#
# with open("test_file_set_sample.pickle", 'wb') as handle:
#     pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
