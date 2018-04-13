import numpy as np
from scipy.stats import pearsonr
def create_movieDir(movies_file):
    # The lines are of the format : movieId,year,movieName
    with open(movies_file) as file:
        dictMovies = {}
        for line in movies_file:
            movieId,year,movieName = line.split(',')
            dictMovies[movieId] = movieName
    return dictMovies

def create_utilityMatrix(ratings_file):
    """
    Computes the total number of unique users and items and returns the values
    """
    #The lines are in the format : movieId,userId,rating
    dictMovies2Index = {}
    dictUsers2Index = {}
    numRatings = 0
    movieIndex = 0
    userIndex = 0
    with open(ratings_file) as file:
        for line in file:
            movieId,userId,rating = line.split(',')
            numRatings += 1
            if userId not in dictUsers2Index:
                dictUsers2Index[userId] = userIndex
                userIndex += 1
            if movieId not in dictMovies2Index:
                dictMovies2Index[movieId] = movieIndex
                movieIndex += 1
    numUsers = userIndex
    numMovies = movieIndex
    print('Summary of {}'.format(ratings_file))
    print('Number of ratings : {}'.format(numRatings))
    print('Number of users : {}'.format(numUsers))
    print('Number of movies : {}\n'.format(numMovies))

    utility_matrix = np.zeros(shape=(numUsers,numMovies))
    with open(ratings_file) as file:
        for line in file:
            movieId,userId,rating = line.split(',')
            user = dictUsers2Index[userId]
            movie = dictMovies2Index[movieId]
            utility_matrix[user][movie] = rating
    return utility_matrix, dictUsers2Index, dictMovies2Index

def evaluate(utility_matrix, dictUsers2Index, dictMovies2Index, testing_file):
    """
    For every user,movie pair in Testing set, find the set of the most similar users in
    the training set who have rated movie and compute the average rating
    """
    predictions = []
    with open(testing_file) as file:
        for line in file:
            movieId, userId, rating = line.split(',')
            print("predicting {}'s rating for {}".format(userId,movieId))
            activeUserIndex = dictUsers2Index[userId]
            activeMovieIndex = dictMovies2Index[movieId]
            # Create a mask for the active user, indicating the rated items
            ratings_activeUser = utility_matrix[activeUserIndex]
            mask = ratings_activeUser != 0
            numSimilarUsers = 0
            sum_similarUserRatings = 0
            for ratings_user in utility_matrix:
                if ratings_user[activeMovieIndex] != 0:
                    # To measure the similarity, we need to only compare the
                    # items rated by both users, so create a mask of common
                    # items
                    currUser_mask = mask * (ratings_user != 0)
                    # if the mask contains only 2 items, pearson correlation
                    # always returns 1 which is not very useful
                    if (sum(currUser_mask) > 2):
                        # Get the ratings of the common items
                        user_commonRatings = ratings_user * currUser_mask
                        activeUser_commonRatings = ratings_activeUser \
                                * currUser_mask
                        pearson_correlation = pearsonr(activeUser_commonRatings,
                                                       user_commonRatings)[0]
                        print(user_commonRatings,activeUser_commonRatings,
                              pearson_correlation)
                        if pearson_correlation > 0.85:
                            print("similar user's rating :\
                                  {}".format(ratings_user))
                            numSimilarUsers += 1
                            sum_similarUserRatings += ratings_user[activeMovieIndex]
            pred_rating = sum_similarUserRatings / numSimilarUsers
            predictions.append(pred_rating)
        print(predictions)
        return predictions

# create_utilityMatrix('netflix-dataset/TrainingRatings.txt')
# create_utilityMatrix('netflix-dataset/TestingRatings.txt')
utility_matrix, dictUsers2Index, dictMovies2Index = create_utilityMatrix('trainingData_fake.txt')
print('utility_matrix')
print(utility_matrix)

create_utilityMatrix('testingData_fake.txt')
evaluate(utility_matrix, dictUsers2Index, dictMovies2Index,
         'testingData_fake.txt')


