from gzip import GzipFile
import json
import datetime
from ann import ANN

TRAIN_FILENAME = "../data/train.json"
TEST_FILENAME  = "../data/test.json"

# features_vector = [user_account_age,
#                    comment_num,
#                    upvote_minus_downvotes,
#                    upvotes_plus_downvotes,
#                    time_of_request
#                    subreddits_at_request # This is a tricky one, think about it later
#                    request_title
#                    request_body
#                    ]
#
# target_vector = [received_pizza (1 => yes, 0 => no)]

class DataHandler:

    #def __init__(self):
    #    self.filename = filename

    def read(self, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            for item in data:
                yield item

    def extract_feature(self, item, type):
            features_vector = []
            target_vector   = []


########### Extract Features

            ## first extract feature that is not text
            ## remember to normalize data

            # preprocess comment_num
            #comment_num = float(item['requester_number_of_comments_at_request'])
            #features_vector.append(comment_num)

            # time_of_request, only care about hour
            # Another hypothesis: check if it is in the morning or at noon, return 0, in the afternoon, return 1
            #time_of_request = int(item['unix_timestamp_of_request'])
            #hour = int(datetime.datetime.fromtimestamp(time_of_request).strftime("%H"))
            #time_of_request = (1 if hour > 16 else 0)
            #features_vector.append(time_of_request)

            # preprocess user_account_age
            user_account_age = float(item['requester_account_age_in_days_at_request'])
            features_vector.append(user_account_age)

            # preprocess upvote_minus_downvotes
            upvote_minus_downvotes = float(item['requester_upvotes_minus_downvotes_at_request'])
            features_vector.append(upvote_minus_downvotes)

            # preprocess upvotes_plus_downvotes
            upvotes_plus_downvotes = float(item['requester_upvotes_plus_downvotes_at_request'])
            features_vector.append(upvotes_plus_downvotes)

            ## Then handle text feature
            ## Don't make the vector too big, maybe :)

            # the interest text list of current user on reddit
            # consider using bag of words on this
            subreddits_at_request = item['requester_subreddits_at_request']
            features_vector.append(len(subreddits_at_request))

            # request title, might need abandon repeated words like [request] etc
            request_title = item['request_title'].split(" ")
            if "return" in request_title:
                request_title = 1
            else:
                request_title = 0
            features_vector.append(request_title)

            # text need to be normalized?
            request_body = item['request_text_edit_aware'].split(" ")

            # request text length is a feature
            features_vector.append(len(request_body))

            # if the requester promised to return the favor
            if ("return" in request_body) or ("pay" in request_body and "back" in request_body):
                return_favor = 1
            else:
                return_favor = 0
            features_vector.append(return_favor)

########### Extract Targets(For training data only)

            if type == "train":
                received_pizza = 1 if item['requester_received_pizza'] == True else 0
                target_vector.append(received_pizza)
                return [features_vector, target_vector]
            else:
                return [features_vector]

    # batch generate data
    def generate_data(self, filename, type="train"):

        data = []

        for item in self.read(filename):
            data_vec = self.extract_feature(item, type)
            #print(data_vec)
            data.append(data_vec)

        feat_vecs = self.normalize([row[0] for row in data])

        if type == "train":
            for i in range(len(data)):
                data[i][0] = feat_vecs[i]
        else:
            data = feat_vecs

        return data

    def normalize(self, data):
        for i in range(len(data[0])):
            column = [vec[i] for vec in data]
            max_val = max(column)

            if max_val == 0: continue

            for j in range(len(data)):
                data[j][i] = data[j][i]/max_val

        return data

    def write_to_result(self, filename, result):
        with open("result.txt", 'w+') as result_file:
            result_file.write("request_id,requester_received_pizza\n")
            for idx, item in enumerate(self.read(filename)):
                result_file.write(item['request_id']+","+str(result[idx])+"\n")

if __name__ == "__main__":

    handler = DataHandler()
    training_data = handler.generate_data(TRAIN_FILENAME)

    print(len(training_data))
    print(training_data[0:10])
    ann = ANN(7,4,1)
    for i in range(20):
        print(i+1)
        ann.train(training_data, 5000)

    testing_data = handler.generate_data(TEST_FILENAME, 'test')
    result = ann.test_without_true_label(testing_data)
    handler.write_to_result(TEST_FILENAME, result)


