import json
def get_api_key(json_path):
    # Specify the path to your JSON file
    file_path = json_path
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def get_ip_port(json_path):
    # Load the JSON data from the file
    with open(json_path, 'r') as file:
        data = json.load(file)

    if not 'ip'in data:
        data['ip'] = "127.0.0.1"
    if not 'port'in data:
        data['port'] =  12009
    return data['ip'], data['port']

def get_num_clients(json_path):
    # Load the JSON data from the file
    with open(json_path, 'r') as file:
        data = json.load(file)
    if not 'num_clients'in data:
        data['num_clients'] = 1000
    return data['num_clients']
    

def close_server_option(json_path):
    # Load the JSON data from the file
    with open(json_path, 'r') as file:
        data = json.load(file)
    if not 'close_server_when_client_die'in data:
        data['close_server_when_client_die'] = False
    return data['close_server_when_client_die']