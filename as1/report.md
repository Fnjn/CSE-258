# CSE 258 Assignment 1 Report

## Purchase Prediction

### Algorithm Description

A simple predictor that return 1 if the user has purchased same category before, Otherwise return 0.

```python
def predictor(u, i):
    try:
        reviewer_pair = reviewer_category_pair[u]
        item_pair = item_category_pair[i]
    except KeyError:
        return 0

    for i in item_pair:
        if i in reviewer_pair: return 1
    return 0
```

### Result
#### Public leaderboard
Ranking:,676/816  Score: 0.63549
#### Private leaderboard
Ranking: 729/816  Score: 0.63414

### Comment
I also experimented K-nearest-neighbors (cosine similarity) and predict based on how many neighbors has purchased this item. But this doesn't give better result than the simple predictor above.

## Rating Prediction

### Algorithm Description

I use Latern Factor model (Matrix Factorization) for this task.


The implementation of matrix factorization is credited to Albert Au Yeung from [his blog post](http://www.albertauyeung.com/post/python-matrix-factorization/)


```python
def mse(self):
    """
    A function to compute the total mean square error
    """
    xs, ys = self.R.nonzero()
    predicted = self.full_matrix()
    error = 0
    for x, y in zip(xs, ys):
        error += pow(self.R[x, y] - predicted[x, y], 2)
    return np.sqrt(error)
```


#### Gradient Descent Update

```python
def sgd(self):
    """
    Perform stochastic graident descent
    """
    for i, j, r in self.samples:
        # Computer prediction and error
        prediction = self.get_rating(i, j)
        e = (r - prediction)

        # Update biases
        self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
        self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

        # Update user and item latent feature matrices
        self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
        self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
```
#### Predictor
1. If the user and item both not in the dataset, return average rating of all items.
2. If user (or item) not in the dataset, return average rating of this item (or user).
3. Otherwise, return the predict rating calculated by
 <img src="rating_predictor_formula.png" alt="Predictor" align="middle" width="200">

```python
def predictor(u, i):
    if u not in RA.user2index and i not in RA.item2index:
        return RA.all_avg
    if u not in RA.user2index:
        return sum([r for u,r in RA.item2user[i]]) / len(RA.item2user[i])
    if i not in RA.item2index:
        return sum([r for u,r in RA.user2item[u]]) / len(RA.user2item[u])

    user_index = RA.user2index[u]
    item_index = RA.item2index[i]
    r = mfa.get_rating(user_index, item_index)
    if r > 5:
        r = 5
    elif r < 0:
        r = 0
    return r
```

### Result
#### Public leaderboard
Ranking: 442/461  Score: 1.19568
#### Private leaderboard
Ranking: 439/461  Score: 1.17019

### Comment
