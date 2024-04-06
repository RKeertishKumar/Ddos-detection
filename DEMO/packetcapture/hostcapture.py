import csv
from scapy.all import sniff, IP

# Callback function to process each captured packet
def process_packet(packet):
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[IP].sport
        dst_port = packet[IP].dport
        timestamp = packet.time

        # Extract more parameters
        ttl = packet[IP].ttl
        protocol = packet[IP].proto
        flags = packet[IP].flags
        length = len(packet)
        
        # Flow duration calculation
        flow_duration = packet.time - packet[0].time

        # Calculate total packet lengths
        fwd_packets_length_total = sum(len(p) for p in packet[IP].payload)
        bwd_packets_length_total = sum(len(p) for p in packet[IP].payload)
        
        # Calculate packet length statistics
        fwd_packet_length_max = max(len(p) for p in packet[IP].payload)
        fwd_packet_length_min = min(len(p) for p in packet[IP].payload)
        fwd_packet_length_mean = sum(len(p) for p in packet[IP].payload) / len(packet[IP].payload)
        fwd_packet_length_std = (sum((len(p) - fwd_packet_length_mean) ** 2 for p in packet[IP].payload) / len(packet[IP].payload)) ** 0.5

        # Specify the output file name
        output_file = 'captured_traffic_loopback.csv'

        # Write packet information to CSV file
        with open(output_file, 'a', newline='') as csv_file:
            fieldnames = ['Timestamp', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port',
                          'TTL', 'Protocol', 'Flags', 'Length', 'Flow Duration', 'Total Fwd Packets', 
                          'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total',
                          'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 
                          'Fwd Packet Length Std']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header row if the file is empty
            if csv_file.tell() == 0:
                writer.writeheader()

            # Write packet information
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
                'Total Fwd Packets': len(packet[IP].payload),
                'Total Backward Packets': len(packet[IP].payload),
                'Fwd Packets Length Total': fwd_packets_length_total,
                'Bwd Packets Length Total': bwd_packets_length_total,
                'Fwd Packet Length Max': fwd_packet_length_max,
                'Fwd Packet Length Min': fwd_packet_length_min,
                'Fwd Packet Length Mean': fwd_packet_length_mean,
                'Fwd Packet Length Std': fwd_packet_length_std
            })

# Start capturing packets from the loopback interface and process each packet using the callback function
print("Capturing loopback traffic...")
sniff(iface='Loopback Pseudo-Interface 1', prn=process_packet, store=0)
