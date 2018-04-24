import json
import requests
import keras

hostname = "http://localhost:9090/api/v1/query?query="
cpu_metrics = ['node_cpu_seconds_total', 'node_cpu_core_throttles_total', 'node_cpu_frequency_hertz, node_cpu_frequency_max_hertz',
                'node_cpu_frequency_min_hertz', 'node_cpu_guest_seconds_total', 'node_cpu_package_throttles_total']

cpu_list = ["0", "1", "2"]

def main():
    for cpu in cpu_list:
        for metric in cpu_metrics:
            url = hostname + metric + '{cpu="'+cpu+'"}'
            print(url)
            r = requests.get(url)
            result = r.json()['data']['result']
            print("result recieved for query: " + str(result))
            print("amount of points: " + str(len(result)))


main()
