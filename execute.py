from recommend_me import *

# Example tags
example_tag = [
    "apple", "banana", "cherry", "dragon", "eclipse", "forest", "galaxy", "horizon",
    "ice", "jungle", "kettle", "lava", "mountain", "nebula", "oasis", "prairie",
    "quartz", "river", "sphinx", "tornado", "unicorn", "volcano", "whisper",
    "xenon", "yacht", "zebra", "amber", "blizzard", "crystal", "diamond",
    "ember", "flame", "glacier", "harbor", "island", "jewel", "kingdom",
    "labyrinth", "meadow", "nightfall", "ocean", "phoenix", "quiver", "rainbow",
    "sapphire", "thunder", "underground", "violet", "waterfall", "zenith"
]


### Gerar dataset aleatório a partir das example_tags
import random

def generate_dataset(random_words):
    data = []
    for id in range(50):
        # Randomly select a number of tags between 5 and 10
        num_tags = random.randint(5, 10)
        # Randomly select tags from the random_words list
        tags = random.sample(random_words, num_tags)
        data.append({"ID": id, "tags": tags})
    return pd.DataFrame(data)

def generate_a_user(raw_tags, n):
    user = dict()
    seed = random.randint(0, 10000)
    findmetags = random.sample(raw_tags, n)
    user['tags'] = findmetags
    user['weights'] = np.arange(len(findmetags), 0, -1)
    return user


if __name__ == "__main__":
    taglist = return_taglist()
    u = generate_a_user(taglist, 5)
    print(u)
    graded_recs = recommend_me(u)
    top_n_rows = graded_recs.nlargest(10, 'score')
    print(top_n_rows[['title','categories','score']])