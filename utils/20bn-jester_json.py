import os
import sys
import json
import pandas as pd


def convert_csv_to_dict(csv_path, subset, max_samples_per_class=sys.maxint):

    data = pd.read_csv(csv_path, header=None)
    keys = []
    key_labels = []
    classes_count = {}
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        slash_rows = str(data.ix[i, 0]).split(';')
        if subset != 'testing':
            class_name = slash_rows[1]
        else:
            class_name = ''

        basename = slash_rows[0]

        if class_name not in classes_count:
            classes_count[class_name] = 0
        if classes_count[class_name] >= max_samples_per_class:
            continue
        
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(class_name)

        classes_count[class_name] += 1
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}
    
    return database

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 0])

    return labels

def convert_20bn_jester_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path, test_csv_path,
                                           max_train_samples, max_val_samples, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training', max_samples_per_class=max_train_samples)
    val_database = convert_csv_to_dict(val_csv_path, 'validation', max_samples_per_class=max_val_samples)
    test_database = convert_csv_to_dict(test_csv_path, 'testing')
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    print ("Database samples: ")
    print ('training', len(train_database.keys()))
    print ('validation', len(val_database.keys()))
    print ('testing', len(test_database.keys()))

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    max_train_samples = int(sys.argv[2])
    max_val_samples = int(sys.argv[3])

    label_csv_path = os.path.join(csv_dir_path, 'jester-v1-labels.csv')
    train_csv_path = os.path.join(csv_dir_path, 'jester-v1-train.csv')
    val_csv_path = os.path.join(csv_dir_path, 'jester-v1-validation.csv')
    test_csv_path = os.path.join(csv_dir_path, 'jester-v1-test.csv')
    dst_json_path = os.path.join(csv_dir_path, '20bn-jester.json')

    convert_20bn_jester_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                                val_csv_path, test_csv_path,
                                                max_train_samples, max_val_samples,
                                                dst_json_path)
