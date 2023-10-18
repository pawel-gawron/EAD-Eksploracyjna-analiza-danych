import sqlite3
import requests
import json
import pandas as pd

conn = sqlite3.connect("/home/pawel/Documents/RISA/sem2/EAD/"
                       "Lab 02 - SQL, RESTful API/chinook-database/"
                       "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite")  # połączenie do bazy danych - pliku
c = conn.cursor()


def Ex1():
    for row in c.execute('SELECT InvoiceId, CustomerId, BillingCity, Total FROM Invoice WHERE BillingCountry LIKE \'USA\' ORDER BY BillingCity DESC'):
        print(f'invoice: {row[0]}, customer: {row[1]}, city: {row[2]}, total: {row[3]}')

def Ex2():
    for row in c.execute('SELECT Name, Title FROM Artist LEFT JOIN Album ON Album.ArtistId = Artist.ArtistId'):
        print(row)

def Ex3():
    req = requests.get("https://blockchain.info/ticker")  # wysłanie zapytania GET pod odpowiedni adres, zapisanie odpowiedzi
    bitcoin_dict = json.loads(req.text)
    print(bitcoin_dict)  # zawartość odpowiedzi znajduje się w polu text

    bitcoin_DataFrame = pd.DataFrame.from_dict(bitcoin_dict, orient='index')
    print(bitcoin_DataFrame)

def Ex4():
    url = "https://api.openweathermap.org/data/2.5/weather"
    api_key = "791f6e3a95a283dc48c6d40abf2d140a"
    latitude = 54.19
    longitude = 19.06
    req = requests.get(f"{url}?lat={latitude}&lon={longitude}&units=metric&exclude=minutely&appid={api_key}")
    print(req.text)

def attack_against(attacker: str, attacked: str, database: pd.DataFrame):

    req = requests.get("https://pokeapi.co/api/v2/pokemon/metapod")  # wysłanie zapytania GET pod odpowiedni adres, zapisanie odpowiedzi
    pokemon_api_dict = json.loads(req.text)
    
    
    # Tworzenie list z danymi
    stats_data = []
    types_data = []

    for stat in pokemon_api_dict["stats"]:
        stat_name = stat["stat"]["name"]
        base_stat = stat["base_stat"]
        stats_data.append((stat_name, base_stat))

    for i, type_entry in enumerate(pokemon_api_dict["types"]):
        type_name = type_entry["type"]["name"]
        if i == 0:
            types_data.append((('type_1', type_name)))
            types_data.append((('type_2', None)))
        else:
            types_data[-1] = (types_data[-1][0], type_name)

    # Tworzenie DataFrame
    columns = ["stat", "value"]
    stats_df = pd.DataFrame(stats_data, columns=columns)
    types_df = pd.DataFrame(types_data, columns=columns)
    pokemon_df = pd.concat([stats_df, types_df], axis=0)

    print(pokemon_df)

    conn = sqlite3.connect("/home/pawel/Documents/RISA/sem2/EAD/Lab 02 - SQL, RESTful API/pokemon_against.sqlite")  # połączenie do bazy danych - pliku
    c = conn.cursor()

    for row in c.execute('SELECT * FROM against_stats WHERE against_bug LIKE \'%against%\''):
        print(row)

    # against_DataFrame = 
        
def Ex5():
    attacker = 'bulbasaur'
    attacked = 'charmander'

    pokedex_DataFrame_history = pd.read_hdf('pokedex_history.hdf5')
    print(pokedex_DataFrame_history)

    attack_against(attacker, attacked, pokedex_DataFrame_history)
    

if __name__ == '__main__':
    Ex1()
    Ex2()
    conn.close()

    Ex3()
    Ex4()
    Ex5()