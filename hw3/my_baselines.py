import gzip
from collections import defaultdict

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

### Rating
def ratingBaseline(ratingPrediction):
    predictions = open("predictions_Rating.txt", 'w')
    for l in open("pairs_Rating.txt"):
        if l.startswith("reviewerID"):
            #header
            predictions.write(l)
            continue
        u,i = l.strip().split('-')
        rating = ratingPrediction(u, i)
        predictions.write(u + '-' + i + ',' + str(rating) + '\n')

    predictions.close()

### Would-purchase
def purchaseBaseline(purchasePrediction):
    predictions = open("predictions_Purchase.txt", 'w')
    for l in open("pairs_Purchase.txt"):
        if l.startswith("reviewerID"):
            #header
            predictions.write(l)
            continue
        u,i = l.strip().split('-')
        purchase = purchasePrediction(u, i)
        predictions.write(u + '-' + i + "," + str(purchase) + "\n")

    predictions.close()

### Rating Evaluation Mean Square Error
def EvaluationRating(ratingPrediction, X, Y):
    mse = 0.
    for (u, i), y in zip(X, Y):
        rating = ratingPrediction(u, i)
        mse += (y - rating) ** 2
    return mse / len(X)

### Purchase Evaluation accuracy
def EvaluationPurchase(purchasePrediction, X, Y):
    acc = 0
    for (u, i), y in zip(X, Y):
        purchase = purchasePrediction(u, i)
        acc += int(y == purchase)
    return acc / len(X)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
  user,business = l['reviewerID'],l['itemID']
  allRatings.append(l['rating'])
  userRatings[user].append(l['rating'])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

def ProfRatingPrediction(u, i):
      if u in userAverage:
        return userAverage[u]
      else:
        return globalAverage


### Would-purchase baseline: just rank which businesses are popular and which are not, and return '1' if a business is among the top-ranked
businessCount = defaultdict(int)
totalPurchases = 0

for l in readGz("train.json.gz"):
  user,business = l['reviewerID'],l['itemID']
  businessCount[business] += 1
  totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPurchases/2: break

def ProfPurchasePrediction(u, i):
      if i in return1:
        return 1
      else:
        return 0
