import numpy as np
import tabulate

def generate_user_vecs(user_vec, num_movies):

    user_vecs = np.tile(user_vec, (num_movies, 1))
    return user_vecs



def predict_movies(sorted_y, sorted_movies, movie_list, maxCount):
    count = 0
    disp = disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(min(maxCount, sorted_y.shape[0])):
        
        movieId = sorted_movies[i, 0].astype(int)
        movie = movie_list[movie_list["movieId"] == movieId]
        if movie.empty:
            continue
        disp.append([np.round(sorted_y[i], 1), movieId, sorted_movies[i, 2],  movie["title"].iloc[0], movie["genres"].iloc[0]])

    table = tabulate.tabulate(disp, tablefmt="html", headers="firstrow")
    return table

def get_user_vecs(userId, user_vecs, movie_vecs, user_to_movies):

    if userId not in user_to_movies:
        print("User Not found")
        return None
    user_vec = None
    for i in range(user_vecs.shape[0]):
        if user_vecs[i, 0] == userId:
            user_vec = user_vecs[i]
            break
    if  user_vec is None:
        print("User not found ")
        return None
    num_items = len(movie_vecs)
    tailed_user_vecs = np.tile(user_vec, (num_items, 1))
    y = np.zeros(num_items)
    rated_movies = user_to_movies[userId]["movies"]

    for i in range(num_items):
        movieId = movie_vecs[i, 0]
        y[i] = rated_movies.get(movieId, 0)

    return tailed_user_vecs, y
def print_existing_user_pred(y_pred, y, user_vecs, movie_vecs, uvs, ivs, movie_list, maxCount):

    count = 0
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]

    for i in range(movie_vecs.shape[0]):
        if y[i, 0] != 0:
            if maxCount == count:
                break
            movieId = movie_vecs[i, 0]
            
            count += 1
            offset = np.nonzero(movie_vecs[i, ivs:] == 1)[0]
            genre_ratings = user_vecs[i, uvs + offset]
            movie = movie_list[movie_list["movieId"] == movieId]
            disp.append([y_pred[i, 0], y[i, 0], user_vecs[i, 0], np.array2string(
                genre_ratings, formatter={"float_kind":lambda x :f"{x:.1f}"},
                separator=",", suppress_small=True),
                        movie_vecs[i, 2], movieId, movie["title"].iloc[0], movie["genres"].iloc[0]])
    table = tabulate.tabulate(disp, tablefmt="html", headers="firstrow")
    return table            
            
def sq_dist(a, b):

    dist = np.sum((a - b)**2)
    return dist