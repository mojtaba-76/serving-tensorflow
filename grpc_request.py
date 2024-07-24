import time
import tensorflow as tf
import grpc
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import numpy as np

def min_max_scale(value, min_value, max_value):
    res = []
    for i in range(len(value)):
        if max_value[i] == min_value[i]:
            res.append(0.5)
        else:
            res.append((value[i] - min_value[i]) / (max_value[i] - min_value[i]))
    return res

def preprocessing(input_data):
    cont_ordered_keys = ['amt_txn', 'sum_amnt', 'max_amnt', 'min_amnt', 'mean_amnt',
                         'cnt_txn', 'std_amnt', 'sum_mcc', 'cnt_mcc', 'cp_sum_amn', 'cp_cnt', 'cnp_sum_amn',
                         'cnp_cnt', 'sum_amn_00', 'cnt_amn_00', 'sum_amn_01',
                         'cnt_amn_01', 'sum_amn_40', 'cnt_amn_40', 'sum_amn_50',
                         'cnt_amn_50', 'avg_msgcrtdate_dur_scnd', 'jalali_month_sin', 'jalali_month_cos', 'jalali_day_sin',
                         'jalali_day_cos', 'hour_sin', 'hour_cos']
        
    rule_ordered_keys = ['is_holiday', 'night', 'weekend', 'month_part', 'free']

    cont = np.array([input_data.get(key, 0) for key in cont_ordered_keys])
    rule = np.array([input_data.get(key, 0) for key in rule_ordered_keys])
    min_values_tensor = np.array([1, 1, 1, 1, 0, 1, 0.0, 1, 1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0,  -0.5000000000000004, 0.8660254037844384, -0.9987165071710528, -0.9948693233918952, -1.0, -1.0])
    max_values_tensor = np.array([900000000000, 6000355833319, 900000000000, 900000000000, 900000000000.0000, 574, 141025376439.84503, 6000355833319, 573, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 86352, -0.5000000000000004, 0.8660254037844384,0.9987165071710528, 0.9795299412524945, 1.0, 1.0])

    cont_new = min_max_scale(cont, min_values_tensor, max_values_tensor)

    input_dict = {
        "cont_input": np.reshape(cont_new, (1, 28)).tolist(),
        "tx_input": np.reshape(input_data["transaction_type"], (1, 5)).tolist(),
        "tr_input": np.reshape(input_data["terminal_type"], (1, 8)).tolist(),
        "mcc_input": np.reshape(input_data["mcc_range"], (1, 5)).tolist(),
        "rule_input": np.reshape(rule, (1, 5)).tolist()
    }

    tx_id = input_data['tx_id']
    return input_dict, tx_id

def is_anomaly(model_output, input_dict):
    cont_input = np.array(input_dict["cont_input"])
    cont_output = np.array(model_output["cont_output"])
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    cont_error = np.mean(np.square(cont_input - cont_output))
    tx_error = cce(input_dict["tx_input"], model_output["tx_output"])
    tr_error = cce(input_dict["tr_input"], model_output["tr_output"])
    mcc_error = cce(input_dict["mcc_input"], model_output["mcc_output"])
    rule_error = bce(input_dict["rule_input"], model_output["rule_output"])

    print(tx_error)
    print(tr_error)
    print(mcc_error)
    print(rule_error)


    total_error_per_sample = (cont_error + tx_error + tr_error + mcc_error + rule_error)
    result = int(total_error_per_sample > 2)
    return result, total_error_per_sample

def categorical_crossentropy_manual(y_true, y_pred, epsilon=1e-7):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  
    log_preds = np.log(y_pred)
    cross_entropy = -np.sum(y_true * log_preds, axis=-1)
    return cross_entropy

def is_anomaly_2(model_output, input_dict):
    cont_input = np.array(input_dict["cont_input"])
    tx_input = np.array(input_dict["tx_input"])
    tr_input = np.array(input_dict["tr_input"])
    mcc_input = np.array(input_dict["mcc_input"])
    rule_input = np.array(input_dict["rule_input"])

    cont_output = np.array(model_output["cont_output"])
    tx_output = np.array(model_output["tx_output"])
    tr_output = np.array(model_output["tr_output"])
    mcc_output = np.array(model_output["mcc_output"])
    rule_output = np.array(model_output["rule_output"])

    # Calculate Mean Squared Error for continuous variables
    cont_error = np.mean(np.square(cont_input - cont_output))

    # Calculate Categorical Crossentropy Error for categorical variables
    tx_error = categorical_crossentropy_manual(tx_input, tx_output)
    tr_error = categorical_crossentropy_manual(tr_input, tr_output)
    mcc_error = categorical_crossentropy_manual(mcc_input, mcc_output)

    # Calculate Binary Crossentropy Error for binary variables
    epsilon=1e-7
    rule_error = -np.mean(rule_input * np.log(rule_output + epsilon) + (1 - rule_input) * np.log(1 - rule_output + epsilon), axis=-1)

    total_error_per_sample = cont_error + tx_error + tr_error + mcc_error + rule_error

    result = int(total_error_per_sample > 2)
    return result, total_error_per_sample



# gRPC channel setup
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

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

# Send the gRPC request
start_time = time.time()
preprocessing_start_time = time.time()
input_dict, tx_id = preprocessing(data)
print(f"preprocessing time: {(time.time() - preprocessing_start_time) * 1000}")

# Create gRPC request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'autoencoder'
request.model_spec.signature_name = 'serving_default'

for key, value in input_dict.items():
    request.inputs[key].CopyFrom(tf.make_tensor_proto(value))

# Send gRPC request and get response
predictions_start_time = time.time()
response = stub.Predict(request)
predictions = {key: tf.make_ndarray(tensor_proto) for key, tensor_proto in response.outputs.items()}
print(f"predictions time: {(time.time() - predictions_start_time) * 1000}")

final_result_start_time = time.time()
final_result, total_error_per_sample = is_anomaly_2(predictions, input_dict)
print(f"final_result time: {(time.time() - final_result_start_time) * 1000}")

print(final_result)
print(total_error_per_sample)

print(f"total time: {(time.time() - start_time) * 1000}")
