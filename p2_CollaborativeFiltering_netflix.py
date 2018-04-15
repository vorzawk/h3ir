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

def compute_rmse(predictions, expected_values):
    return np.sqrt(((predictions - expected_values) ** 2).mean())

def compute_mae(predictions, expected_values):
    return (np.absolute(predictions - expected_values)).mean()

def compute_cosineSimilarity(a,b):
    return np.dot(a,b)/ (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity(u, v):
    u_m = u[u != 0].mean()
    v_m = v[v != 0].mean()
    u_mask = (u != 0) * u_m
    v_mask = (v != 0) * v_m
    return compute_cosineSimilarity(u - u_mask, v - v_mask)

def evaluate(utility_matrix, dictUsers2Index, dictMovies2Index, testing_file):
    """
    For every user,movie pair in Testing set, find the set of the most similar users in
    the training set who have rated movie and compute the average rating
    """
    def get_commonRatings(ratings, mask):
        common_ratings = ratings * mask
        # Remove the zero elements, these lead to a higher correlation value
        # than the reality
        return common_ratings[common_ratings != 0]

    predictions = []
    expected_values = []
    import sys
    sys.stdout.flush()
    with open(testing_file) as file:
        for line in file:
            stripped_line = line.strip() # Remove leading/trailing whitespace
            movieId, userId, rating = stripped_line.split(',')
            expected_values.append(float(rating))
            print("predicting {}'s rating for {}".format(userId,movieId))
            activeUserIndex = dictUsers2Index[userId]
            activeMovieIndex = dictMovies2Index[movieId]
            # Create a mask for the active user, indicating the rated items
            ratings_activeUser = utility_matrix[activeUserIndex]
            mask = ratings_activeUser != 0
            numSimilarUsers = 0
            sum_similarUserRatings = 0
            # Initialize the predicted rating with the average rating value for
            # the active user, later this is adjusted with the collaborative
            # value
            pred_rating = ratings_activeUser[ratings_activeUser != 0].mean()
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
                        user_commonRatings = get_commonRatings(
                            ratings_user, currUser_mask)
                        activeUser_commonRatings = get_commonRatings(
                            ratings_activeUser, currUser_mask)
                        #pearson_correlation = pearsonr(activeUser_commonRatings, user_commonRatings)[0]
                        cosine_similarity = compute_similarity(user_commonRatings,
                                           activeUser_commonRatings)
                      #  print(user_commonRatings, activeUser_commonRatings,
                      #        pearson_correlation)
                        if cosine_similarity > 0.7:
                            print("active user and similar user's rating : {} and {}".format(activeUser_commonRatings, user_commonRatings))
                            numSimilarUsers += 1
                            sum_similarUserRatings += ( ratings_user[activeMovieIndex] - user_commonRatings.mean())
            pred_rating += (sum_similarUserRatings / (numSimilarUsers + 0.0001))
            predictions.append(pred_rating)
        predictions = np.array(predictions)
        expected_values = np.array(expected_values)
        print(predictions, expected_values)
        return predictions, expected_values

training_file = 'netflix-dataset/TrainingRatings.txt'
testing_file = 'netflix-dataset/TestingRatings_small.txt'
# training_file = 'trainingData_fake.txt'
# testing_file = 'testingData_fake.txt'
#testing_file = 'netflix-dataset/TestingRatings.txt'
utility_matrix, dictUsers2Index, dictMovies2Index = create_utilityMatrix(training_file)
#print('utility_matrix')
#print(utility_matrix)

# ans = compute_cosineSimilarity(np.array([1,2,0,0]), np.array([-0.5,0.5,0,0]))
# print(ans)
# ans = compute_similarity(np.array([1,2,0,0]), np.array([-0.5,0.5,0,0]))
# print(ans)
#create_utilityMatrix('testingData_fake.txt')
predictions, expected_values = evaluate(utility_matrix, dictUsers2Index, dictMovies2Index, testing_file)
rmse = compute_rmse(predictions, expected_values)
print("rmse : {}".format(rmse))
mae = compute_mae(predictions, expected_values)
print("mae : {}".format(mae))

