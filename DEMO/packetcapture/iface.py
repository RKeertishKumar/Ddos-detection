from scapy.arch.windows import get_windows_if_list

# Get a list of available network interfaces
iface_list = get_windows_if_list()

# Print information about each interface
for iface in iface_list:
    print(f"Interface Name: {iface['name']}")
    print(f"Description: {iface['description']}")
    print("="*50)
