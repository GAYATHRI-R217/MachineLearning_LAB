import csv

with open('play_tennis.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

# Remove header
data = data[1:]

# Initial hypothesis
h = ['0', '0', '0', '0', '0', '0']

k = 0

# FIRST PRINT (as in your output)
print("for the training sample 0")
print("the hypothesis is:", [h], end=" ")

k = 1

for row in data:
    print("for the training sample", k)

    if row[-1] == "True":
        for j in range(len(h)):
            if h[j] == '0':
                h[j] = row[j]
            elif h[j] != row[j]:
                h[j] = '?'

    print("the hypothesis is:", [h], end=" ")
    k += 1

print("\nthe maximally specific hypothesis is", [h])