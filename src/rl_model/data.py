import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from rl_model.config.utils import set_global_seed

class Data(): 

    def __init__(self,
                 seed = None):
        self.path = "../data/"
        self.eq1 = self.path+"equipment1.csv"
        self.eq2 = self.path+"equipment2.csv"
        self.response = self.path+"response.csv"
        self.important_features =  ['sensor_10', 'sensor_11', 'sensor_12', 'sensor_17', 'sensor_18','sensor_19', 'sensor_20', 'sensor_21', 'sensor_23', 'sensor_24','sensor_29', 'sensor_3', 'sensor_36', 'sensor_37', 'sensor_39','sensor_41', 'sensor_47', 'sensor_48', 'sensor_54', 'sensor_55','sensor_7', 'sensor_8']
        self.test_size = 0.3
        self.SEED = seed

    def load_data(self,path):
        data = pd.read_csv(path,encoding='latin-1',sep=";")
        return data

    def prepare_sensor_data(self,eq1,eq2):        
        # inner join of eq1 and eq2
        sensor_data = pd.merge(eq1,eq2,on=['lot','wafer','timestamp'],how='inner',sort=False).reset_index(drop=True)     
        # check if every wafer has a timestamp length of 176 - this is nessessary because the neural network needs always the same shape 
        sensor_data = sensor_data.groupby(['lot']).filter(lambda x: x['wafer'].count()%176==0)
        # change datatype of lot/wafers to string
        sensor_data[["lot", "wafer"]] = sensor_data[["lot", "wafer"]].astype('str')

        # normalize the dataset with MinMaxScaler
        colnames_sensors = sensor_data.iloc[:,3:].columns
        sensor_data[colnames_sensors] = MinMaxScaler().fit_transform(sensor_data[colnames_sensors])

        return sensor_data

    def prepare_response_data(self,response):
        # RESPONSE 
        # drop duplicates (every result/row appears 2 times - this is incorrect and needs to be cleaned)
        response = response.drop_duplicates().reset_index(drop=True)
        response[["lot", "wafer"]] = response[["lot", "wafer"]].astype('str')
        # translate classes: good = 1 and bad = 2
        response['class'] = response['class'].map(lambda x: 1 if x == 'good' else 2)

        return response

    def split_good_bad(self,sensor_data,response,class_good_bad):
        """
        The split between good and bad is necessary because in the train/test split the same 
        proportion of good and bad wafers is needed.
        """
         # sensor data + response (good + bad)
        sensor_response = pd.merge(sensor_data, response,  how='left', left_on=['lot','wafer'], right_on = ['lot','wafer']).drop(["response"],axis=1).reset_index(drop=True)

        # good/bad sensor data
        good_bad_sensor_data = sensor_response[sensor_response['class']==class_good_bad].drop(["class"],axis=1)

        # only response data (with unique lots/wafer)
        response_data  = sensor_response[['lot','wafer','class']].drop_duplicates().reset_index(drop=True)
        good_bad_response_data = response_data[response_data['class']==class_good_bad].reset_index(drop=True)

        return good_bad_sensor_data, good_bad_response_data

    def split_data_train_test(self,sensor_data, response_data):
        # SHAPE FOR NEURAL NETWORK
        # create 3D Matrix 
        sequences = sensor_data.groupby(['lot','wafer'])
        sequences = np.array(sequences[self.important_features].apply(lambda x : x.values.tolist()).tolist())

        labels = np.array(response_data['class'])

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=self.test_size, random_state=self.SEED)

        return x_train, x_test, y_train, y_test

    def combine_good_bad_data(self,x_train_good, x_test_good, y_train_good, y_test_good,x_train_bad, x_test_bad, y_train_bad, y_test_bad):
        x_train = np.append(x_train_good,x_train_bad,axis=0)
        x_test = np.append(x_test_good,x_test_bad,axis=0)
        y_train = np.append(y_train_good,y_train_bad,axis=0)
        y_test = np.append(y_test_good,y_test_bad,axis=0)

        return x_train, x_test, y_train, y_test

    def shuffle_data(self,x,y):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x_data = x[randomize]
        y_data = y[randomize]

        return x_data, y_data

    def run(self):
        set_global_seed(self.SEED)
        # load data
        eq1 = self.load_data(self.eq1)
        eq2 = self.load_data(self.eq2)
        resp = self.load_data(self.response)

        # build sensor data
        sensor_data = self.prepare_sensor_data(eq1,eq2)

        # build response data
        response = self.prepare_response_data(resp)

        # split data in good and bad wafer
        good_sensor_data, good_response_data  = self.split_good_bad(sensor_data,response,class_good_bad=1)
        bad_sensor_data, bad_response_data = self.split_good_bad(sensor_data,response,class_good_bad=2)

        # split data in train and test data
        x_train_good, x_test_good, y_train_good, y_test_good = self.split_data_train_test(good_sensor_data, good_response_data)
        x_train_bad, x_test_bad, y_train_bad, y_test_bad = self.split_data_train_test(bad_sensor_data, bad_response_data) 

        # bring good and bad wafers together
        x_train, x_test, y_train, y_test = self.combine_good_bad_data(x_train_good, x_test_good, y_train_good, y_test_good,x_train_bad, x_test_bad, y_train_bad, y_test_bad)
        
        # shuffle train and test data (x and y in the same way because now the bad wafer are all in the end of the array)
        x_train, y_train = self.shuffle_data(x_train, y_train)
        x_test, y_test = self.shuffle_data(x_test, y_test)

        return x_train, x_test, y_train, y_test