import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('play_teennis.csv')

# Convert to numpy
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

print("Output")
print("Instances are:")
print(concepts)

print("Target Values are:", target)

def learn(concepts, target):

    print("Initialization of specific_h and genearal_h")

    specific_h = concepts[0].copy()
    print("Specific Boundary:", specific_h)

    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]

    print("Generic Boundary:", general_h)

    for i, h in enumerate(concepts):

        print("Instance", i+1, "is", h)

        if target[i] == "yes":
            print("Instance is Positive")

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        elif target[i] == "no":
            print("Instance is Negative")

            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Specific Bundary after", i+1, "Instance is", specific_h)
        print("Generic Boundary after", i+1, "Instance is", general_h)

    # Remove rows with all '?'
    general_h = [g for g in general_h if g != ['?', '?', '?', '?', '?', '?']]

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specific_h:")
print(s_final)

print("Final General_h:")
print(g_final)