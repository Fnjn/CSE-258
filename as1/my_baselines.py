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
    pos_cnt = 0
    total_cnt = 0
    for l in open("pairs_Purchase.txt"):
        if l.startswith("reviewerID"):
            #header
            predictions.write(l)
            continue
        u,i = l.strip().split('-')
        purchase = purchasePrediction(u, i)
        if purchase == 1:
            pos_cnt += 1
        total_cnt += 1
        predictions.write(u + '-' + i + "," + str(purchase) + "\n")

    print(pos_cnt / total_cnt, total_cnt)
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
