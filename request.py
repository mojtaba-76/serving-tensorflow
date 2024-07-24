import time
import json
import requests
import tensorflow as tf

def min_max_scale(value, min_value, max_value):
    res = []
    for i in range(len(value)):
        if max_value[i] == min_value[i]:
            res.append(0.5)
        else:
            res.append((value[i] - min_value[i]) / (max_value[i] - min_value[i]))
    return res

def preprocessing(input_data):
    cont_ordered_keys = ['amt_txn', 'sum_amnt', 'max_amnt','min_amnt', 'mean_amnt',
                         'cnt_txn','std_amnt', 'sum_mcc','cnt_mcc','cp_sum_amn','cp_cnt','cnp_sum_amn',
                         'cnp_cnt','sum_amn_00','cnt_amn_00', 'sum_amn_01',
                         'cnt_amn_01','sum_amn_40','cnt_amn_40','sum_amn_50',
                         'cnt_amn_50','avg_msgcrtdate_dur_scnd','jalali_month_sin','jalali_month_cos','jalali_day_sin',
                         'jalali_day_cos','hour_sin','hour_cos']
        
    rule_ordered_keys = ['is_holiday','night','weekend','month_part','free']

    cont = [input_data.get(key, 0) for key in cont_ordered_keys]
    rule = [input_data.get(key, 0) for key in rule_ordered_keys]
    min_values_tensor = [1, 1, 1, 1, 0, 1, 0.0, 1, 1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0,  -0.5000000000000004, 0.8660254037844384, -0.9987165071710528, -0.9948693233918952, -1.0, -1.0]
    max_values_tensor = [900000000000, 6000355833319, 900000000000, 900000000000, 900000000000.0000, 574, 141025376439.84503, 6000355833319, 573, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 86352, -0.5000000000000004, 0.8660254037844384,0.9987165071710528, 0.9795299412524945, 1.0, 1.0]

    rule_tensor = tf.constant(rule, dtype=tf.float32)
    tx_tensor = tf.constant(input_data["transaction_type"], dtype=tf.float32)
    tr_tensor = tf.constant(input_data["terminal_type"], dtype=tf.float32)
    mcc_tensor = tf.constant(input_data["mcc_range"], dtype=tf.float32)

    cont_new = min_max_scale(cont, min_values_tensor, max_values_tensor)
    cont_tensor = tf.constant(cont_new, dtype=tf.float32)

    input_dict = {
        "cont_input": tf.reshape(cont_tensor, (28)).numpy().tolist(),
        "tx_input": tf.reshape(tx_tensor, (5)).numpy().tolist(),
        "tr_input": tf.reshape(tr_tensor, (8)).numpy().tolist(),
        "mcc_input": tf.reshape(mcc_tensor, (5)).numpy().tolist(),
        "rule_input": tf.reshape(rule_tensor, (5)).numpy().tolist()
    }

    tx_id = input_data['tx_id']
    return input_dict, tx_id

def is_anomaly(model_output, input_dict):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    cont_error = mse(input_dict["cont_input"], model_output["cont_output"])
    tx_error = cce(input_dict["tx_input"], model_output["tx_output"])
    tr_error = cce(input_dict["tr_input"], model_output["tr_output"])
    mcc_error = cce(input_dict["mcc_input"], model_output["mcc_output"])
    rule_error = bce(input_dict["rule_input"], model_output["rule_output"])

    total_error_per_sample = (cont_error + tx_error + tr_error + mcc_error + rule_error)
    result = int(total_error_per_sample > 2)
    return result, total_error_per_sample

# Define the URL where TensorFlow Serving is running
url = 'http://localhost:8501/v1/models/autoencoder:predict'

# Prepare the data for prediction
data = {
    "amt_txn": 1300000.0, "sum_amnt": 1300000.0, "max_amnt": 1300000.0, "min_amnt": 1300000.0, "mean_amnt": 1300000.0,
    "cnt_txn": 1.0, "std_amnt": 0.0, "sum_mcc": 1300000.0, "cnt_mcc": 0.0, "cp_sum_amn": 0.0, "cp_cnt": 0.0,
    "cnp_sum_amn": 0.0, "cnp_cnt": 0.0, "sum_amn_00": 0.0, "cnt_amn_00": 0.0, "sum_amn_01": 0.0, "cnt_amn_01": 0.0,
    "sum_amn_40": 0.0, "cnt_amn_40": 0.0, "sum_amn_50": 0.0, "cnt_amn_50": 0.0, "avg_msgcrtdate_dur_scnd": 0.0,
    "jalali_month_sin": 0.5, "jalali_month_cos": 0.8660254, "jalali_day_sin": -0.89780456, "jalali_day_cos": -0.44039416,
    "hour_sin": -0.25881904, "hour_cos": -0.9659258, "transaction_type": [0.0, 0.0, 0.0, 0.0, 0.0],
    "terminal_type": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "mcc_range": [0.0, 0.0, 1.0, 0.0, 0.0],
    "is_holiday": 0.0, "free": 0.0, "night": 1.0, "weekend": 0.0, "month_part": 0.0, "tx_id": "14031266710"
}

# Send the POST request
start_time = time.time()
preprocessing_start_time = time.time()
input_dict, tx_id = preprocessing(data)
print(f"preprocessing time: {(time.time() - preprocessing_start_time) * 1000}")
payload = {
    "signature_name": "serving_default",
    "instances": [input_dict]
}
predictions_start_time = time.time()
response = requests.post(url, json=payload)
predictions = response.json()
print(f"predictions time: {(time.time() - predictions_start_time) * 1000}")
final_result_start_time = time.time()
final_result, total_error_per_sample = is_anomaly(predictions['predictions'][0],input_dict)
print(f"final_result time: {(time.time() - final_result_start_time) * 1000}")
print(f"total time: {(time.time() - start_time) * 1000}")
