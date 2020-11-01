from configparser import ConfigParser

parser = ConfigParser()

parser['setting'] = {
    'batch_size': '32',
    'epochs': '20'
}

with open('config.ini', 'w') as file:
    parser.write(file)


