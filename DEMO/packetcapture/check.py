from scapy.all import *

def process_packet(packet):
    print(packet.summary())

# Use the default loopback interface name
sniff(prn=process_packet, store=0)
