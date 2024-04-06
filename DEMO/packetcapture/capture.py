import csv
from scapy.all import sniff, IP

# Global variables to store flow information
flows = {}

# Callback function to process each captured packet
def process_packet(packet):
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[IP].sport
        dst_port = packet[IP].dport
        timestamp = packet.time
        length = len(packet)
        
        # Extract more parameters if available
        ttl = packet[IP].ttl
        protocol = packet[IP].proto
        flags = packet[IP].flags
        
        # Calculate flow duration and update flow information
        flow_key = (src_ip, src_port, dst_ip, dst_port)
        if flow_key not in flows:
            flows[flow_key] = {'start_time': timestamp, 'fwd_packets': 0, 'bwd_packets': 0,
                               'fwd_packets_length_total': 0, 'bwd_packets_length_total': 0,
                               'fwd_packets_length_max': 0, 'fwd_packets_length_min': float('inf')}
        
        flow = flows[flow_key]
        flow_duration = timestamp - flow['start_time']

        # Update flow packet count based on packet direction
        flow['fwd_packets'] += 1
        flow['fwd_packets_length_total'] += length
        flow['fwd_packets_length_max'] = max(flow['fwd_packets_length_max'], length)
        flow['fwd_packets_length_min'] = min(flow['fwd_packets_length_min'], length)
        
        # Write packet and flow information to CSV file
        with open('captured_traffic.csv', 'a', newline='') as csv_file:
            fieldnames = ['Timestamp', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port',
                          'TTL', 'Protocol', 'Flags', 'Length', 'Flow Duration', 'Total Fwd Packets',
                          'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total',
                          'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                          'Fwd Packet Length Std']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header row if the file is empty
            if csv_file.tell() == 0:
                writer.writeheader()

            # Calculate mean and standard deviation for forward packets length
            fwd_packets_length_mean = flow['fwd_packets_length_total'] / (flow['fwd_packets'] + 1)  # Add 1 to avoid division by zero
            fwd_packets_length_std = sum((len(pkt) - fwd_packets_length_mean) ** 2 for pkt in packet) ** 0.5 / len(packet) if flow['fwd_packets'] > 1 else 0
            
            # Write packet and flow information
            writer.writerow({
                'Timestamp': timestamp,
                'Source IP': src_ip,
                'Destination IP': dst_ip,
                'Source Port': src_port,
                'Destination Port': dst_port,
                'TTL': ttl,
                'Protocol': protocol,
                'Flags': flags,
                'Length': length,
                'Flow Duration': flow_duration,
                'Total Fwd Packets': flow['fwd_packets'],
                'Total Backward Packets': flow['bwd_packets'],
                'Fwd Packets Length Total': flow['fwd_packets_length_total'],
                'Bwd Packets Length Total': flow['bwd_packets_length_total'],
                'Fwd Packet Length Max': flow['fwd_packets_length_max'],
                'Fwd Packet Length Min': flow['fwd_packets_length_min'],
                'Fwd Packet Length Mean': fwd_packets_length_mean,
                'Fwd Packet Length Std': fwd_packets_length_std
            })

# Start capturing packets and process each packet using the callback function
print("Capturing traffic...")
sniff(prn=process_packet, store=0)
