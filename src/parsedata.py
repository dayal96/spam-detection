#!/usr/bin/python3.6
import glob


# Read the ionosphere data for analysis.
def read_sms_spam_data():
    data = []
    Y = []

    with open("data/sms-spam-collection-dataset/spam.csv") as f:
        # Ignore the first line
        line = f.readline()
        line = f.readline()
        while line:
            line = line.strip().split(",")
            data.append(preprocess("".join(line[1:])))

            if line[0] == 'ham':
                Y.append(0)
            else:
                Y.append(1)

            line = f.readline()
    
    return (data, Y)

# Read the room occupancy prediction data for analysis.
def read_tweet_spam_data():
    filename = "data/spam-or-not-spam-dataset/spam_or_not_spam.csv"
    data = []
    Y = []

    with open(filename) as f:
        f.readline()
        line = f.readline()
        while line:
            line = line.strip().split(",")
            data.append(line[0])
            Y.append(int(line[1]))
            line = f.readline()
    
    return (data, Y)

# Read the diabetic retinopathy data for analysis.
def read_ling_spam():
    folder = "data/ling-spam-bare/part"
    data = []
    Y = []

    for i in range(1,11):
        files = glob.glob(folder + str(i) + "/*")

        for filename in files:
            with open(filename) as f:
                line = f.read()
                data.append(preprocess(line))
                if "spmsg" in filename:
                    Y.append(1)
                else:
                    Y.append(0)
    
    return (data, Y)


def preprocess(text):
    text = " ".join(text.split("\n"))
    text = [x.strip().lower() for x in text.split(" ")]
    text = filter(remove_empties, text)
    text = " ".join(text)
    return text

def remove_empties(string):
    return (len(string) > 1)

