import numpy as np
import csv

def transformFile(train_array, test_array, train_file_name, test_file_name):
    new_columns = ['FileName',
                   'Atelectasis',
                   'Cardiomegaly',
                   'Effusion',
                   'Infiltration',
                   'Mass',
                   'Nodule',
                   'Pneumonia',
                   'Pneumothorax']

    with open('./Data_Entry_2017.csv') as f:
        with open(train_file_name, "w+") as wf_train:
            with open(test_file_name, "w+") as wf_test:
                writer_train = csv.writer(wf_train)
                writer_test = csv.writer(wf_test)

                writer_train.writerow(new_columns)
                writer_test.writerow(new_columns)

                lines = f.read().splitlines()
                del lines[0]

                for i in range(len(lines)):
                    data = []
                    row = lines[i].split(',')

                    data.append(row[0])
                    for col in new_columns[1:]:
                        if col in row[1]:
                            data.append(1)
                        else:
                            data.append(0)

                    temp = np.array(data[1:])
                    if row[1] == 'No Finding' or sum(temp == 1) > 0 :
                        if data[0] in test_array:
                            writer_test.writerow(data)
                        else:
                            writer_train.writerow(data)

def getTxtContent(txt_file):
    result = []
    with open(txt_file) as file:
        result = file.read().splitlines()
    return result


train_txt = getTxtContent('./train_val_list.txt')
test_txt = getTxtContent('./test_list.txt')

transformFile(train_txt, test_txt, './Train_Label.csv', './Test_Label.csv')

