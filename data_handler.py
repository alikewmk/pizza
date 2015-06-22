from gzip import GzipFile
import json
import re
from ann import ANN

TRAIN_FILENAME = "../data/train.json"
TEST_FILENAME  = "../data/test.json"

# features_vector = [user_account_age,
#                    comment_num,
#                    upvote_minus_downvotes,
#                    upvotes_plus_downvotes,
#                    first_post_time,
#                    subreddits_at_request,
#                    request_body_length,
#                    reciprocity
#                    has_evidence
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

            ## first extract social factor features
            ## remember to normalize data

            # preprocess user_account_age
            user_account_age = float(item['requester_account_age_in_days_at_request'])
            features_vector.append(user_account_age)

            # preprocess comment_num
            comment_num = float(item['requester_number_of_comments_in_raop_at_request'])
            features_vector.append(comment_num)

            # preprocess upvote_minus_downvotes
            upvote_minus_downvotes = float(item['requester_upvotes_minus_downvotes_at_request'])
            features_vector.append(upvote_minus_downvotes)

            # preprocess upvotes_plus_downvotes
            upvotes_plus_downvotes = float(item['requester_upvotes_plus_downvotes_at_request'])
            features_vector.append(upvotes_plus_downvotes)

            # The time of the first post requester made
            first_post_time = float(item['requester_days_since_first_post_on_raop_at_request'])
            features_vector.append(first_post_time)

            ## Then handle text feature
            ## Don't make the vector too big, maybe :)

            # text need to be normalized?
            request_body = item['request_text_edit_aware'].split(" ")
            # request text length is a feature
            features_vector.append(len(request_body))

            # reciprocity
            if re.search(r'pay it back|pay it forward|return the favor|give back', item['request_text_edit_aware'], re.IGNORECASE):
                return_favor = 1
            else:
                return_favor = 0
            features_vector.append(return_favor)

            # politeness
            #if re.search(r'appreciate|hello|hey|great|awesome|please|could|would|thank', item['request_text_edit_aware'], re.IGNORECASE):
            #    politeness = 1
            #else:
            #    politeness = 0
            #features_vector.append(politeness)
            #

            # If provide evidence in request
            if re.search(r'\.jpg|\.gif|\.png', item['request_text_edit_aware']):
                has_image = 1
            else:
                has_image = 0
            features_vector.append(has_image)

            # the interest text list of current user on reddit
            # consider using bag of words on this
            subreddits_at_request = item['requester_subreddits_at_request']
            features_vector.append(len(subreddits_at_request))

########### Extract Target Labels(For training data only)

            if type == "train":
                received_pizza = 1 if item['requester_received_pizza'] == True else 0
                target_vector.append(received_pizza)
                return [features_vector, target_vector]
            else:
                return [features_vector]

    # 1.count of words with sentiment information in message
    # 2.generate bag of words vec for sentiment information
    # see which one is better
    def words_count(self, dict, words):

        dict = ["appreciate", "hello", "hey", "please", "could", "would", "thank", "thanks"]
        pattern = re.comiple(join(dict, "|"))
        for item in self.read(filename):
            request_body = item['request_text_edit_aware'].split(" ")
            request_title = item['']
        return len(re.findall(r'pay it back|pay it forward|return',"return return return"))

    # batch generate data
    def generate_data(self, filename, type="train"):

        data = []

        for item in self.read(filename):
            data_vec = self.extract_feature(item, type)
            #print(data_vec)
            data.append(data_vec)

        feat_vecs = self.normalize_by_max([row[0] for row in data])

        if type == "train":
            for i in range(len(data)):
                data[i][0] = feat_vecs[i]
        else:
            data = feat_vecs

        return data

    def normalize_by_max(self, data):
        for i in range(len(data[0])):
            column = [vec[i] for vec in data]
            max_val = max(column)

            if max_val == 0: continue

            for j in range(len(data)):
                data[j][i] = data[j][i]/max_val

        return data

    # not as good as normalized by max
    def normalize_by_mean(self, data):
        for i in range(len(data[0])):
            column = [vec[i] for vec in data]
            mean = st.mean(column)
            sd   = st.variance(column)**0.5

            for j in range(len(data)):
                data[j][i] = (data[j][i] - mean)/sd

        return data

    def write_to_result(self, filename, result):
        with open("result.txt", 'w+') as result_file:
            result_file.write("request_id,requester_received_pizza\n")
            for idx, item in enumerate(self.read(filename)):
                result_file.write(item['request_id']+","+str(result[idx])+"\n")

if __name__ == "__main__":

    handler = DataHandler()
    data = handler.generate_data(TRAIN_FILENAME)

    ann = ANN(9,10,10,10,1)

    for i in range(20):
        print(i+1)
        ann.train(data, 5000)

    testing_data = handler.generate_data(TEST_FILENAME, 'test')
    result = ann.test_without_true_label(testing_data, 0.23)
    handler.write_to_result(TEST_FILENAME, result)

    training_data = data[:3500]

    testing_data = data[3500:]
    result = ann.test(testing_data, 0.23)
