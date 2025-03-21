import os
import json

folder = 'HypoBench-code/citation_prediction_data/radiology_2017_2022'
train_data = json.load(open(os.path.join(folder, 'citation_train.json')))
test_data = json.load(open(os.path.join(folder, 'citation_test.json')))
valid_data = json.load(open(os.path.join(folder, 'citation_val.json')))

for i, data in enumerate([train_data, test_data, valid_data]):
    for i in range(len(data['label'])):
        if data['label'][i] == "0":
            data['label'][i] = "unimpactful"
        elif data['label'][i] == "1":
            data['label'][i] = "impactful"
        # else:
        #     raise ValueError

        
json.dump(train_data, open(os.path.join(folder, 'citation_train.json'), 'w'))
json.dump(test_data, open(os.path.join(folder, 'citation_test.json'), 'w'))
json.dump(valid_data, open(os.path.join(folder, 'citation_val.json'), 'w'))

