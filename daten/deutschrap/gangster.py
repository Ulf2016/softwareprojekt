
import json

path ="gangster_rap.json"

with open(path, "r", encoding='utf-8') as read_file:
    data = json.load(read_file)

for artist in data['total']:
    for song in artist['songs']:
        for word in song['lyrics'].split():
            print(word)

