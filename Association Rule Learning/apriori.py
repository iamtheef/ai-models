import pandas as pd
from apyori import apriori

# header option s[ecifies that there are no headers in the data file (aka including the first row)
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []

# removing nan values from the dataset & making a new array 'transactions'
for i in range(0, len(dataset)):
    transactions.append([dataset.values[i, j] for j in range(0, 20)
                         if not pd.isnull(dataset.values[i, j])])

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2,
                min_lift=3, min_length=2, max_length=2)


# Displaying the first results coming directly from the output of the apriori function
list_rules = list(rules)

# Putting the results well organised into a Pandas DataFrame


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsinDataFrame = pd.DataFrame(inspect(list_rules), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Displaying the results non sorted
resultsinDataFrame

# Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))
