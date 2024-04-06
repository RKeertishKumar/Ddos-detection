import csv

# Function to extract additional parameters and create headers
def extract_additional_params(data):
    additional_params = set()
    for row in data:
        info = row[-1]
        info_parts = info.split(' ')
        for part in info_parts:
            if '=' in part:
                param, _ = part.split('=')
                additional_params.add(param)
    return list(additional_params)

# Function to split the "Info" column and include additional parameters as headers
def split_info_column(data, additional_params):
    new_data = []
    for row in data:
        line = row[-1]  # Extract the last element containing "Info"
        parts = row[:-1]  # Exclude the last element from splitting
        info_parts = line.split(' ')  # Split the "Info" column
        additional_values = {param: None for param in additional_params}  # Initialize all additional parameters to None
        for part in info_parts:
            if '=' in part:
                param, value = part.split('=')
                additional_values[param] = value  # Set the parameter to its value
        parts.extend([additional_values[param] for param in additional_params])  # Add the additional parameter values to the original data
        new_data.append(parts)
    return new_data

# Function to write data to CSV file
def write_to_csv(data, filename, header):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # Write header
        writer.writerow(header)
        # Write data rows
        for line in data:
            writer.writerow(line)

# Read data from wireshark_data.csv file, skipping the first row containing headers
with open('wireshark_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)  # Skip the first row (header)
    data = list(reader)

# Extract additional parameters
additional_params = extract_additional_params(data)

# Generate header with additional parameters
initial_header = ['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
header = initial_header + additional_params

# Remove initial header from header list
header.remove('Info')

# Split the "Info" column and include additional parameters as headers
split_data = split_info_column(data, additional_params)

# Calculate flow characteristics
flows = {}
for row in split_data:
    # Extract flow parameters
    src, dst, protocol = row[2], row[3], row[4]
    flow_key = (src, dst, protocol)
    
    # Update flow data
    if flow_key not in flows:
        flows[flow_key] = {'total_packets': 0, 'total_bytes': 0, 'duration': 0,
                           'fwd_packets_length_total': 0, 'bwd_packets_length_total': 0,
                           'fwd_packets_length_max': 0, 'fwd_packets_length_min': float('inf')}
    flows[flow_key]['total_packets'] += 1
    flows[flow_key]['total_bytes'] += int(row[5])  # Assuming Length is the total bytes
    flows[flow_key]['duration'] = max(float(row[1]), flows[flow_key]['duration'])

    # Update packet length statistics
    packet_length = int(row[5])  # Assuming Length is the total bytes
    if packet_length > 0:
        if src == flow_key[0]:
            # Forward packet
            flows[flow_key]['fwd_packets_length_total'] += packet_length
            flows[flow_key]['fwd_packets_length_max'] = max(packet_length, flows[flow_key]['fwd_packets_length_max'])
            flows[flow_key]['fwd_packets_length_min'] = min(packet_length, flows[flow_key]['fwd_packets_length_min'])

# Append flow data to each packet's row
for row in split_data:
    src, dst, protocol = row[2], row[3], row[4]
    flow_key = (src, dst, protocol)
    flow_data = flows[flow_key]
    row.extend([flow_data['total_packets'], flow_data['total_bytes'], flow_data['duration'],
                flow_data['fwd_packets_length_total'], flow_data['bwd_packets_length_total'],
                flow_data['fwd_packets_length_max'], flow_data['fwd_packets_length_min']])

# Append flow data headers to the header list
header.extend(['Total Packets (Flow)', 'Total Bytes (Flow)', 'Flow Duration',
               'Fwd Packets Length Total', 'Bwd Packets Length Total',
               'Fwd Packet Length Max', 'Fwd Packet Length Min'])

# Write processed data to processed_data.csv file
write_to_csv(split_data, 'processed_data.csv', header)
