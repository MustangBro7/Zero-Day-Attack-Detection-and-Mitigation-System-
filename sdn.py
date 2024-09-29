import sys
import joblib
import numpy as np
import pandas as pd
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
import requests
import time
from threading import Thread
import traceback
from sklearn.ensemble import RandomForestClassifier
import os

# Increase the recursion limit to avoid RecursionError
sys.setrecursionlimit(10000)

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.model_path = 'random_forest_multi_attack_model.pkl'
        self.server_url = 'http://10.0.2.15:5000'  # Server URL
        self.scaler = joblib.load('feature_scaler_all_features.pkl')  # Load the initial scaler
        self.model = None  # Ensure that model is initialized as None at start
        self.enable_prediction = False  # Flag to enable/disable prediction during control traffic

        # Load or create a new RandomForest model
        self.update_model_from_server()  # Fetch the global model from server on startup

        # Features used in the model
        self.features = [
            'Fwd Header Length', 'Fwd Packet Length Std', 'Bwd Packets/s', 'Fwd Packet Length Mean',
            'Bwd Header Length', 'Fwd IAT Mean', 'Packet Length Std', 'Flow IAT Std',
            'Min Packet Length', 'Fwd Packet Length Min', 'Avg Fwd Segment Size', 'Max Packet Length',
            'ACK Flag Count', 'Packet Length Variance', 'Packet Length Mean', 'Bwd Packet Length Max',
            'Bwd IAT Std', 'Flow IAT Mean', 'Bwd IAT Mean', 'Average Packet Size', 'Bwd IAT Total'
        ]

        self.local_training_data = []
        self.training_interval = 60  # Train every 60 seconds
        self.update_interval = 60  # Check for updates every 60 seconds
        self.last_trained = time.time()
        self.last_updated = time.time()

        # Start a thread to periodically check for global model updates
        self.start_update_model_thread()

    def start_update_model_thread(self):
        """
        Periodically fetch the global model from the server.
        """
        def update_model_loop():
            while True:
                if time.time() - self.last_updated > self.update_interval:
                    self.update_model_from_server()  # Fetch model from the server
                    self.last_updated = time.time()
                time.sleep(10)  # Check for updates every 10 seconds

        update_thread = Thread(target=update_model_loop)
        update_thread.daemon = True
        update_thread.start()

    def update_model_from_server(self):
        """
        Fetch the global model IPFS CID from the server and download it from IPFS.
        """
        try:
            self.logger.info("Fetching global model CID from the server...")
            response = requests.get(f'{self.server_url}/get_model')
            if response.status_code == 200:
                ipfs_cid = response.json().get('ipfs_cid')
                if ipfs_cid:
                    self.logger.info(f"Fetching model from IPFS using CID: {ipfs_cid}")
                    os.system(f"ipfs cat {ipfs_cid} > {self.model_path}")
                    self.model = joblib.load(self.model_path)
                    self.logger.info("Model successfully updated from IPFS.")
                else:
                    self.logger.warning("No IPFS CID found in server response.")
                    self.model = RandomForestClassifier(n_estimators=100)  # Initialize a fresh model if not found
            else:
                self.logger.warning(f"Failed to fetch model CID from the server. Status code: {response.status_code}")
                self.model = RandomForestClassifier(n_estimators=100)  # Initialize a fresh model if server fetch fails
        except Exception as e:
            self.logger.error(f"Failed to update the local model from server: {e}")
            self.model = RandomForestClassifier(n_estimators=100)  # Initialize fresh model on failure
            traceback.print_exc()

    def train_local_model(self):
        """
        Trains the local RandomForest model on the collected data and sends the model to the global server.
        """
        try:
            if len(self.local_training_data) > 0:
                self.logger.info(f"Starting local model training with {len(self.local_training_data)} samples.")

                # Create a DataFrame from local training data
                local_df = pd.DataFrame(self.local_training_data, columns=self.features + ['Label'])
                X_local = local_df[self.features]
                y_local = local_df['Label']

                # Scale the features
                self.logger.info("Scaling features...")
                X_local_scaled = self.scaler.fit_transform(X_local)

                # Train the RandomForest model locally
                self.logger.info("Fitting the local RandomForest model with the scaled data...")
                self.model.fit(X_local_scaled, y_local)

                # Save the local model
                joblib.dump(self.model, self.model_path)
                self.logger.info("Local model training complete. Sending the model to the global server...")

                # Send the trained model to the global server
                self.send_model_to_server()

                # Clear local training data after training
                self.local_training_data = []
            else:
                self.logger.info("No new data to train the local model.")
        except Exception as e:
            self.logger.error(f"Error during local model training: {e}")
            traceback.print_exc()

    def send_model_to_server(self):
        """
        Sends the trained local model to the global server for aggregation.
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_data = f.read()

            response = requests.post(f'{self.server_url}/update_model', files={'model': model_data})

            if response.status_code == 200:
                self.logger.info("Local model sent to the global server successfully.")
            else:
                self.logger.error(f"Failed to send model to the global server. Status Code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to send model to the global server: {e}")
            traceback.print_exc()

    # Function to distinguish control traffic from data traffic
    def is_data_traffic(self, tcp_pkt, udp_pkt):
        return tcp_pkt or udp_pkt  # Returns True only if TCP/UDP packets are present

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

        # Enable predictions after initial OpenFlow setup
        self.enable_prediction = True  # Enable prediction after the control setup

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, priority=priority,
                                    match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    def extract_features(self, pkt, tcp_pkt, udp_pkt):
        features = [0] * len(self.features)

        if tcp_pkt:
            # Extract features for TCP packets
            features[0] = float(tcp_pkt.window_size)  # Fwd Header Length
            features[1] = float(np.std([len(pkt)]))  # Fwd Packet Length Std
            features[2] = 1  # Bwd Packets/s (placeholder)
            features[3] = float(len(pkt))  # Fwd Packet Length Mean
            features[4] = 1  # Bwd Header Length (placeholder)
            features[5] = float(np.mean([len(pkt)]))  # Fwd IAT Mean
            features[6] = float(np.std([len(pkt)]))  # Packet Length Std
            features[7] = float(np.std([len(pkt)]))  # Flow IAT Std
            features[8] = min(len(pkt), tcp_pkt.dst_port)  # Min Packet Length
            features[9] = len(pkt)  # Fwd Packet Length Min
            features[10] = len(pkt)  # Avg Fwd Segment Size
            features[11] = max(len(pkt), tcp_pkt.dst_port)  # Max Packet Length
            features[12] = 1  # ACK Flag Count (placeholder)
            features[13] = float(np.var([len(pkt)]))  # Packet Length Variance
            features[14] = float(np.mean([len(pkt)]))  # Packet Length Mean
            features[15] = max(len(pkt), tcp_pkt.dst_port)  # Bwd Packet Length Max
            features[16] = float(np.std([len(pkt)]))  # Bwd IAT Std
            features[17] = float(np.mean([len(pkt)]))  # Flow IAT Mean
            features[18] = float(np.mean([len(pkt)]))  # Bwd IAT Mean
            features[19] = float(np.mean([len(pkt)]))  # Average Packet Size
            features[20] = len(pkt)  # Bwd IAT Total

        elif udp_pkt:
            # Extract features for UDP packets
            features[3] = float(udp_pkt.length) if hasattr(udp_pkt, 'length') else float(len(pkt))  # Fwd Packet Length Mean
            features[0] = float(udp_pkt.src_port) if hasattr(udp_pkt, 'src_port') else 0  # Fwd Header Length
            features[1] = float(np.std([len(pkt)]))  # Fwd Packet Length Std
            features[2] = 1  # Bwd Packets/s (placeholder)
            features[4] = 1  # Bwd Header Length (placeholder)
            features[5] = float(np.mean([len(pkt)]))  # Fwd IAT Mean
            features[6] = float(np.std([len(pkt)]))  # Packet Length Std
            features[7] = float(np.std([len(pkt)]))  # Flow IAT Std
            features[8] = udp_pkt.dst_port if hasattr(udp_pkt, 'dst_port') else 0  # Min Packet Length
            features[9] = features[3]  # Fwd Packet Length Min
            features[10] = features[3]  # Avg Fwd Segment Size
            features[11] = max(len(pkt), udp_pkt.dst_port) if hasattr(udp_pkt, 'dst_port') else 0  # Max Packet Length
            features[12] = 1  # ACK Flag Count (placeholder)
            features[13] = float(np.var([len(pkt)]))  # Packet Length Variance
            features[14] = float(np.mean([len(pkt)]))  # Packet Length Mean
            features[15] = max(len(pkt), udp_pkt.dst_port) if hasattr(udp_pkt, 'dst_port') else 0  # Bwd Packet Length Max
            features[16] = float(np.std([len(pkt)]))  # Bwd IAT Std
            features[17] = float(np.mean([len(pkt)]))  # Flow IAT Mean
            features[18] = float(np.mean([len(pkt)]))  # Bwd IAT Mean
            features[19] = float(np.mean([len(pkt)]))  # Average Packet Size
            features[20] = len(pkt)  # Bwd IAT TotalSize

        return features

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        # Check if it's actual data traffic and if prediction is enabled
        if self.enable_prediction and self.is_data_traffic(tcp_pkt, udp_pkt):
            features = self.extract_features(pkt, tcp_pkt, udp_pkt)

            try:
                features_df = pd.DataFrame([features], columns=self.features)
                features_scaled = self.scaler.transform(features_df)

                # Check if the shape is consistent
                # self.logger.info(f"Scaled features shape: {features_scaled.shape}")

                # Ensure features_scaled is a 2D array
                if features_scaled.ndim == 1:
                    features_scaled = features_scaled.reshape(1, -1)

                prediction = self.model.predict(features_scaled)[0]
                print(prediction)
                if prediction != 0:  # Non-normal traffic (attack)
                    self.logger.info("Attack detected from %s", src)
                    self.logger.info("Blocking traffic from %s", src)
                else:
                    self.logger.info("Normal traffic detected from %s", src)

                # Store data for local training
                self.local_training_data.append(features + ['Attack' if prediction != 0 else 'Normal'])

                # Train the local model periodically
                if time.time() - self.last_trained > self.training_interval:
                    self.train_local_model()
                    self.last_trained = time.time()

            except Exception as e:
                self.logger.error(f"Error in feature scaling or prediction: {e}")
                traceback.print_exc()
        else:
            self.logger.info("Normal traffic detected from %s", src)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
