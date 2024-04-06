import csv
import os
import pyshark

def capture_traffic(interface, output_file):
    # Create a CSV file and write the header
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                      'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max',
                      'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                      'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                      'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
                      'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
                      'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                      'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
                      'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
                      'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max',
                      'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                      'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                      'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size',
                      'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                      'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                      'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
                      'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
                      'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
                      'Idle Max', 'Idle Min', 'Label', 'Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Capture network traffic
        capture = pyshark.LiveCapture(interface=interface)

        # Process each packet
        for packet in capture.sniff_continuously():
            # Extract packet information and write to CSV
            row = {}
            for field in fieldnames:
                try:
                    row[field] = packet[field]
                except KeyError:
                    row[field] = ''

            writer.writerow(row)

if __name__ == "__main__":
    interface = "Adapter for loopback traffic capture"  # Update this with your interface name
    output_file = "network_traffic.csv"  # Output CSV file
    capture_traffic(interface, output_file)
