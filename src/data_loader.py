from datasets import load_dataset

def load_tweeteval_dataset():
    """
    Upload TweetEval dataset, sentiment analysis.
    Dataset splits: train, validation e test.
    """
    dataset = load_dataset("sentiment140", split="train")
    return dataset

if __name__ == "__main__":
    dataset = load_tweeteval_dataset()
    print(dataset)
