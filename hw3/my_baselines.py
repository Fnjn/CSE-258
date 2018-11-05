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
