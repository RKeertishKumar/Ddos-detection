import pyshark

def extract_flow_values(packet):
    # Extract basic flow information
    src_ip = packet.ip.src
    dst_ip = packet.ip.dst
    src_port = packet[pkt.transport_layer].srcport
    dst_port = packet[pkt.transport_layer].dstport
    protocol = packet.transport_layer
    
    # Calculate flow duration
    flow_duration = packet.sniff_time.timestamp() - packet[0].sniff_time.timestamp()
    
    # Check if the packet is a forward or backward packet
    if src_ip == "127.0.0.1" and src_port == "5000":  # Define your source IP and port
        fwd_packets = 1
        bwd_packets = 0
    elif dst_ip == "127.0.0.1" and dst_port == "5000":  # Define your destination IP and port
        fwd_packets = 0
        bwd_packets = 1
    else:
        fwd_packets = 0
        bwd_packets = 0
    
    return flow_duration, fwd_packets, bwd_packets

# Example usage
capture = pyshark.LiveCapture(interface='Adapter for loopback traffic capture')
for packet in capture.sniff_continuously():
    flow_duration, fwd_packets, bwd_packets = extract_flow_values(packet)
    
    print(f"Flow Duration: {flow_duration}, Total Fwd Packets: {fwd_packets}, Total Backward Packets: {bwd_packets}")
