
import json
import operator

files = ['gangster_rap.json', 'gangster_rap_2.json', 'gangster_rap_3.json', 'gangster_rap_4.json']

for f in files:
    with open(f, "r") as read_file:
        data = json.load(read_file)

    for artist in data['total']:
        print(artist['artist'])

