from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import threading

app = Flask(__name__)

model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

global_model_path = os.path.join(model_dir, 'random_forest_multi_attack_model.pkl')
ipfs_cid_file = os.path.join(model_dir, 'ipfs_cid.txt')

# Lock to prevent simultaneous access to model
model_lock = threading.Lock()

# Load the global model if it exists, otherwise create a new one
if os.path.exists(global_model_path):
    global_model = joblib.load(global_model_path)
else:
    global_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize empty model

# Function to get the current IPFS CID of the global model
def get_current_ipfs_cid():
    if os.path.exists(ipfs_cid_file):
        with open(ipfs_cid_file, 'r') as f:
            return f.read().strip()
    return None

# Function to upload the global model to IPFS and return the CID
def upload_model_to_ipfs():
    try:
        os.system(f"ipfs add {global_model_path} > {model_dir}/ipfs_output.txt")
        with open(f"{model_dir}/ipfs_output.txt", 'r') as f:
            output = f.readlines()
            for line in output:
                if 'added' in line:
                    cid = line.split()[1]
                    with open(ipfs_cid_file, 'w') as cid_file:
                        cid_file.write(cid)
                    return cid
    except Exception as e:
        print(f"Error uploading model to IPFS: {e}")
    return None

@app.route('/get_model', methods=['GET'])
def get_model():
    """
    Endpoint to provide the current global model CID to nodes.
    """
    try:
        current_cid = get_current_ipfs_cid()
        if current_cid:
            return jsonify({'ipfs_cid': current_cid}), 200
        else:
            # If the file exists but is empty, try uploading the model and updating the CID
            if os.path.exists(global_model_path):
                new_cid = upload_model_to_ipfs()
                if new_cid:
                    return jsonify({'ipfs_cid': new_cid}), 200
                else:
                    return jsonify({'error': 'Failed to upload model to IPFS'}), 500
            return jsonify({'error': 'No model available on IPFS'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_model', methods=['POST'])
def update_model():
    """
    Endpoint to receive a model from a node and aggregate it with the global model.
    """
    try:
        if 'model' in request.files:
            node_model_data = request.files['model'].read()

            # Save the node's model temporarily
            node_model_path = os.path.join(model_dir, 'node_rf_model.pkl')
            with open(node_model_path, 'wb') as f:
                f.write(node_model_data)

            # Load the node's model
            node_model = joblib.load(node_model_path)

            # Acquire a lock to avoid simultaneous model updates
            with model_lock:
                # Aggregate the node model with the global model
                aggregate_model_trees(node_model)

                # Save the updated global model
                joblib.dump(global_model, global_model_path)

                # Upload the aggregated global model to IPFS
                new_cid = upload_model_to_ipfs()
                if new_cid:
                    # Print the new CID to the server logs
                    print(f"Model updated and uploaded to IPFS. New CID: {new_cid}")
                    return jsonify({'message': 'Model updated and uploaded to IPFS', 'ipfs_cid': new_cid}), 200
                else:
                    return jsonify({'error': 'Failed to upload model to IPFS'}), 500
        else:
            return jsonify({'error': 'No model file found in request'}), 400
    except Exception as e:
        print(f"Error receiving model update: {e}")
        return jsonify({'error': str(e)}), 500


def aggregate_model_trees(node_model):
    """
    Aggregates the node model trees with the global model trees.
    Combines the decision trees from each RandomForest.
    """
    global global_model

    try:
        if global_model is None or len(global_model.estimators_) == 0:
            global_model = node_model
        else:
            # Combine trees by appending node's estimators to global model's estimators
            global_model.estimators_ += node_model.estimators_

            # If we want to limit the number of trees in the global model, we can trim it:
            max_trees = 100  # Example: Limit to 100 trees total
            if len(global_model.estimators_) > max_trees:
                global_model.estimators_ = global_model.estimators_[:max_trees]

            print(f"Global model now has {len(global_model.estimators_)} trees.")
    except Exception as e:
        print(f"Error during model aggregation: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
