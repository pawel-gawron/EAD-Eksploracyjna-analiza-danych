import sqlite3
import requests
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime

def attack_against(attacker: str, attacked: str, database: pd.DataFrame):

    vec = [attacker, attacked]

    for pokemon_iter, pokemon in enumerate(vec):
        req = requests.get("https://pokeapi.co/api/v2/pokemon/" + pokemon)  # wysłanie zapytania GET pod odpowiedni adres, zapisanie odpowiedzi
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

        if pokemon_iter == 0:
            pokemon_attacker_df = pd.concat([stats_df, types_df], axis=0).reset_index(drop=True)

            print("Attacker: " + pokemon + "\n", pokemon_attacker_df)
        else:
            pokemon_attacked_df = pd.concat([stats_df, types_df], axis=0).reset_index(drop=True)

            print("Attacked: " + pokemon + "\n", pokemon_attacked_df)

    type_1_attacker = pokemon_attacker_df[pokemon_attacker_df['stat'] == 'type_1']['value']
    type_2_attacker = pokemon_attacker_df[pokemon_attacker_df['stat'] == 'type_2']['value']
    merged_column_attacker = pd.concat([type_1_attacker, type_2_attacker], axis=0)
    count_non_none_values_attacker = np.sum(merged_column_attacker.notna())

    type_1_attacked = pokemon_attacked_df[pokemon_attacked_df['stat'] == 'type_1']['value']
    type_2_attacked = pokemon_attacked_df[pokemon_attacked_df['stat'] == 'type_2']['value']
    merged_column_attacked = pd.concat([type_1_attacked, type_2_attacked], axis=0)
    count_non_none_values_attacked = np.sum(merged_column_attacked.notna())

    conn = sqlite3.connect("/home/pawel/Documents/RISA/sem2/EAD - Eksploracyjna analiza danych/Lab 02 - SQL, RESTful API/pokemon_against.sqlite")  # połączenie do bazy danych - pliku
    c = conn.cursor()

    against_dict = {}

    for i in range(count_non_none_values_attacked):
        column_name = f'against_{merged_column_attacked.values[i]}'
        pokemon_name = f'{attacker.capitalize()}'
        row = c.execute(f"SELECT {column_name} FROM against_stats WHERE against_stats.name = '{pokemon_name}'").fetchone()
        value = row[0]
        against_dict[merged_column_attacked.values[i]] = value

    idx = None
    value = None
    if len(against_dict.keys()) == 2:
        if list(against_dict.values())[0] == list(against_dict.values())[1]:
            rand = random.randint(0, 1)
            idx = rand
            value = list(against_dict.values())[idx]
        else:
            value = max(against_dict, key=against_dict.get)
    else:
        value = list(against_dict.values())[0]

    print("value: ", value)

    result = value*(pokemon_attacker_df[pokemon_attacker_df['stat'] == 'attack']['value'].values[0] +
                    pokemon_attacker_df[pokemon_attacker_df['stat'] == 'special-attack']['value'].values[0] +
                    pokemon_attacker_df[pokemon_attacker_df['stat'] == 'speed']['value'].values[0]/2) -(
                    pokemon_attacker_df[pokemon_attacked_df['stat'] == 'hp']['value'].values[0] +
                    pokemon_attacker_df[pokemon_attacked_df['stat'] == 'defense']['value'].values[0] +
                    pokemon_attacker_df[pokemon_attacked_df['stat'] == 'special-defense']['value'].values[0] +
                    pokemon_attacker_df[pokemon_attacked_df['stat'] == 'speed']['value'].values[0]/3)
    print('result:\n', result)

    if result <= 0:
        return False
    else:
        return True

def pokedex():
    attacker = 'bulbasaur'
    attacked = 'arcanine'

    pokedex_DataFrame_history = pd.read_hdf('pokedex_history.hdf5')

    now = datetime.now()

    if attack_against(attacker, attacked, pokedex_DataFrame_history):
        new_data = {'meeting_date': [now], 'name': [attacked.capitalize()]}
        
        pokedex_DataFrame_history = pd.concat([pokedex_DataFrame_history, pd.DataFrame(new_data, index=[0])], ignore_index=True)

        pokedex_DataFrame_history.to_hdf('pokedex_history.hdf5', key='history', mode='w')

    print(pokedex_DataFrame_history)

if __name__ == '__main__':
    pokedex()