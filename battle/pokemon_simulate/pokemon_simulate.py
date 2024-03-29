"""
Code edited from:
https://github.com/tylerwmarrs/pokemon-battle-simulation
"""

from multiprocessing import Pool, cpu_count
import os
import random

from math import floor, sqrt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, '../data')

# load in type modifiers, pokemon, etc
POKEMON_MOVES = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'pokemon_moves_detailed.csv'))
POKEMON_STATS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'pokemon_stats_no_weirdos.csv'))
TYPE_MODS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'type_modifiers.csv')).set_index('attack_type')
TYPES_DICT = {
    "normal": 1,
    "fighting": 2,
    "flying": 3,
    "ground": 4,
    "rock": 5,
    "bug": 6,
    "poison": 7,
    "ghost": 8,
    "fire": 9,
    "water": 10,
    "grass": 11,
    "electric": 12,
    "ice": 13,
    "psychic": 14,
    "dragon": 15,
    "fairy": 16,
    "steel": 17,
    "dark": 18
}
INVERSE_TYPES_DICT = inv_map = {v: k for k, v in TYPES_DICT.items()}
POKEMON_AVAIL = set(list(POKEMON_STATS['pokemon'].unique()))
VERBOSE = False
VERBOSE_COUNT = True
NUM_SIMULATIONS = 1000


class Move(object):
    """
    Encapsulates a move to apply for a given pokemon. It is used to keep track
    of the power points available, damage and move type.
    """
    def __init__(self):
        self.current_pp = None

    def get_id(self):
        return int(self.url.split('/')[-2])
    
    def __str__(self):
        out = ['Name: {}'.format(self.name), 'Type: {}'.format(self.type)]

        for k, v in self.__dict__.items():
            if k in ('name', 'type', 'url', 'pokemon') or v is None:
                continue
            
            out.append('{}: {}'.format(k, v))
        
        return '\n'.join(out)


class Pokemon(object):
    """
    Encapsulates the moves and stats of a Pokemon for battle. We also use it
    to keep track of the current hit points and moves.
    """
    def __init__(self, name, level):
        self.name = name.lower()
        self.level = level
        self.types = []
        self.current_hp = None
        
        self.all_moves = []
        self.moves = []
        self.attack = None
        self.defense = None
        self.hp = None
        self.special_attack = None
        self.special_defense = None
        self.speed = None
        
        if self.name not in POKEMON_AVAIL:
            raise RuntimeError('{} is not available!'.format(self.name))
        
        self.__load_stats()
        self.__load_moves()
        self.pick_moves()
    
    def __load_stats(self):
        query = POKEMON_STATS['pokemon'] == self.name
        if query.sum() != 1:
            raise RuntimeError('{} expecting 1 result for stats, got {}'.format(self.name, query.sum()))
        
        stats = POKEMON_STATS[query].iloc[0].to_dict()
        
        IV = 0
        EV = 0

        self.hp = floor(((stats['hp'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + self.level + 10
        self.current_hp = self.hp
        self.attack = floor(((stats['attack'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + 5
        self.defense = floor(((stats['defense'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + 5
        self.special_attack = floor(((stats['special-attack'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + 5
        self.special_defense = floor(((stats['special-defense'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + 5
        self.speed = floor(((stats['speed'] + IV) * 2 + floor(sqrt(EV) / 4.0)) * self.level / 100) + 5

        self.types = [TYPES_DICT[x] for x in stats['types'].split(',')]
        if len(self.types) == 1:
            self.types.append(0)
        
    def __load_moves(self):
        query = POKEMON_MOVES['pokemon'] == self.name
        if query.sum() < 1:
            raise RuntimeError('{} has no moves!'.format(self.name))
        
        for idx, row in POKEMON_MOVES[query].drop_duplicates().iterrows():
            move = Move()
            dict_row = row.to_dict()
            for k, v in dict_row.items():
                val = v
                if isinstance(val, float) and np.isnan(val):
                    val = None

                if isinstance(val, str) and val.strip() == '':
                    val = None

                setattr(move, k.replace('move_', ''), val)
            move.current_pp = move.pp
            move.type = TYPES_DICT[move.type]

            # Make dragon type moves normal
            if move.type == 15:
                move.type = 1
            self.all_moves.append(move)
    
    def pick_moves(self):
        # only pick damaging moves
        damage_moves = []
        for move in self.all_moves:
            if 'damage' in move.category:
                damage_moves.append(move)
        
        # enable fewer than 4 moves to be randomly chosen
        max_moves = 4
        if len(damage_moves) < max_moves:
            max_moves = len(damage_moves)                                
        
        self.moves = np.random.choice(damage_moves, max_moves, replace=False)
        
        self.has_moves = True
        if len(self.moves) < 1:
            self.has_moves = False
            
    def reset(self):
        self.current_hp = self.hp
        
        for move in self.moves:
            move.current_pp = move.pp

    def __str__(self):
        move_str = []
        for move in self.moves:
            move_str.append('{} - {}'.format(move.name, move.type))
            
        return f"""
        =================
        Pokemon: {self.name}
        =================
        Level:         {self.level}
        Types:         {self.types}
        HP:            {self.current_hp}
        Speed:         {self.speed}
        Attack:        {self.attack}
        Defense:       {self.defense}
        Sp. Attack:    {self.special_attack}
        Sp. Defense:   {self.special_defense}
        =====
        Moves
        =====
        {[(move.name, move.current_pp, move.pp) for move in self.moves]}
        """


def is_critical_hit(base_speed, move_crit_rate):
    """
    TODO: amping abilities not applied - focus energy etc..
    this is bugged in Gen 1 - but we will just ignore it.
    """
    prob = base_speed / 512
    if move_crit_rate == 1:
        prob = base_speed / 64
    
    chance = np.random.rand()
    return chance <= prob


def apply_move(self, attacker, defender, move):
    """
    Applies a damaging move attacker->defender where the attacker and defenders
    are instances of Pokemon. The move is an instance of the move being
    applied.
    """
    # determine if it is a critical hit or not
    is_crit = is_critical_hit(attacker.speed, move.crit_rate)

    # determine if move applied is the same type as the pokemon or not
    # when it is the same, a 1.5x bonus is applied
    # STAB = same type attack bonus
    stab = 1
    if move.type in attacker.types:
        stab = 1.5

    # determine the move damage class to figure out attack/def stats to use
    attack = attacker.attack
    defense = defender.defense
    if move.damage_class == 'special':
        attack = attacker.special_attack
        defense = defender.special_defense

    # grab type modifier
    modifier = 1
    try:
        attack_type = move.type.title()
        for dtype in defender.types:
            modifier *= TYPE_MODS.loc[attack_type][dtype.title()]
    except:
        pass

    # NOTE: attacker level is hard coded to 10
    # level = 10
    power = move.power
    if power is None:
        power = 1

    damage = 1
    if move.name == 'seismic-toss':
        damage = attacker.level
    else:
        damage = calculate_damage(attacker.level, attack, power, defense, stab, modifier, is_crit)

    # compute number of times to apply the move
    times_to_apply = 1
    if move.min_hits and move.max_hits:
        times_to_apply = np.random.choice(np.arange(move.min_hits, move.max_hits + 1))

    damage *= times_to_apply

    # apply damage to pokemon and reduce move pp
    defender.current_hp -= damage
    move.current_pp -= 1

    if VERBOSE:
        print('{} damaged {} with {} for {} hp'.format(
            attacker.name,
            defender.name,
            move.name,
            damage
        ))
        print('{} pp for {} is {}/{}'.format(attacker.name, move.name, move.current_pp, move.pp))
        print('{} hp is {}/{}'.format(defender.name, defender.current_hp, defender.hp))


def calculate_damage(a, b, c, d, x, y, crit):
    """
    a = attacker's Level
    b = attacker's Attack or Special
    c = attack Power
    d = defender's Defense or Special
    x = same-Type attack bonus (1 or 1.5)
    y = Type modifiers (40, 20, 10, 5, 2.5, or 0)
    z = a random number between 217 and 255
    crit = true or false
    """
    z = np.random.choice(np.arange(217, 256))
    crit_mult = 2
    if crit:
        crit_mult = 4
    
    damage = np.floor(((((crit_mult * a) / 5) + 2) * b * c) / d)
    damage = np.floor(damage / 50) + 2
    damage = np.floor(damage * x)    
    damage = np.floor(damage * y)
    if y > 0 and damage == 0:
        damage = 1
    
    # This is how you could compute the minimum and maximum
    # damage that this ability will do. Only for reference.
    # min_damage = np.floor((damage * 217) / 255)
    # max_damage = np.floor((damage * 255) / 255)
    
    return np.floor((damage * z) / 255)

def is_battle_winnable(attacker, defender):
    n_zeros = 0
    for move in attacker.moves:
        attack_type = move.type
        attack_type_label = INVERSE_TYPES_DICT[attack_type].capitalize()
        for defender_type in defender.types:
            if defender_type != 0:
                defender_type_label = INVERSE_TYPES_DICT[defender_type].capitalize()
                if attack_type_label in TYPE_MODS.columns and defender_type_label in TYPE_MODS.columns:
                    modifier = TYPE_MODS[defender_type_label][attack_type_label]
                    if modifier == 0:
                        n_zeros += 1
    if n_zeros < floor(len(attacker.moves) / 2.0):
        return True
    return False



def apply_move(attacker, defender, move):
    """
    Applies a damaging move attacker->defender where the attacker and defenders
    are instances of Pokemon. The move is an instance of the move being
    applied.
    """
    # determine if it is a critical hit or not
    is_crit = is_critical_hit(attacker.speed, move.crit_rate)
    
    # determine if move applied is the same type as the pokemon or not
    # when it is the same, a 1.5x bonus is applied
    # STAB = same type attack bonus
    stab = 1

    if move.type in attacker.types:
        stab = 1.5
    
    # determine the move damage class to figure out attack/def stats to use
    attack = attacker.attack
    defense = defender.defense
    if move.damage_class == 'special':
        attack = attacker.special_attack
        defense = defender.special_defense
    
    # grab type modifier
    modifier = 1
    attack_type = move.type
    attack_type_label = INVERSE_TYPES_DICT[attack_type].capitalize()
    for defender_type in defender.types:
        if defender_type != 0:
            defender_type_label = INVERSE_TYPES_DICT[defender_type].capitalize()
            if attack_type_label in TYPE_MODS.columns and defender_type_label in TYPE_MODS.columns:
                modifier *= TYPE_MODS[defender_type_label][attack_type_label]
    print('MODIFIER:', modifier)

    power = move.power
    if power is None:
        power = 1
    
    damage = 1
    if move.name == 'seismic-toss':
        damage = attacker.level
    else:
        damage = calculate_damage(attacker.level, attack, power, defense, stab, modifier, is_crit)
    
    # compute number of times to apply the move
    times_to_apply = 1
    if move.min_hits and move.max_hits:
        times_to_apply = np.random.choice(np.arange(move.min_hits, move.max_hits + 1))
    
    damage *= times_to_apply
    
    # apply damage to pokemon and reduce move pp
    defender.current_hp -= damage

    #############################################
    # REMOVING PP DECREASE FOR TRAINING PURPOSE #
    #############################################
    move.current_pp -= 1
    
    if VERBOSE:
        print('{} damaged {} with {} for {} hp'.format(
            attacker.name,
            defender.name,
            move.name,
            damage
        ))
        print('{} pp for {} is {}/{}'.format(attacker.name, move.name, move.current_pp, move.pp))
        print('{} hp is {}/{}'.format(defender.name, defender.current_hp, defender.hp))


def random_move(pokemon):
    """
    Naive and exhaustive approach in choosing a move. It does not pick the most
    optimal move; only random. We also ensure that there is enough PP.
    """
    iters = 0
    move = None
    while move is None and iters < 100:
        move = np.random.choice(pokemon.moves)
        if move.current_pp < 1 or move is None:
            move = None
        
        iters += 1
    
    return move


def is_critical_hit(base_speed, move_crit_rate):
    """
    TODO: amping abilities not applied - focus energy etc..
    this is bugged in Gen 1 - but we will just ignore it.
    """
    prob = base_speed / 512
    if move_crit_rate == 1:
        prob = base_speed / 64

    chance = np.random.rand()
    return chance <= prob


def choose_move(pokemon):
    return random_move(pokemon)
        
        
def simulate_battle(pokemon, pokemon_b):
    """
    Simulates a single battle against the two provided Pokemon instances. The
    battle is concluded when a pokemon loses all of their HP or both pokemon
    run out of PP for all of their moves.
    
    Pokemon are reset at the beginning of each battle. A reset consists of:
    
    1. Random damaging moves selected (up to 4)
    2. PP for moves being restored
    3. HP being reset
    
    We also randomly choose which pokemon attacks first with an equal chance.
    """
    stats = {
        'pokemon': pokemon.name,
        'pokemonb': pokemon_b.name,
        'moves': 0,
        'winner': None,
        'first_attack': None
    }
    
    pokemon.reset()
    pokemon_b.reset()

    ## Describe state
    
    if pokemon.speed > pokemon_b.speed:
        start = 'a'
        stats['first_attack'] = pokemon.name
    if pokemon.speed < pokemon_b.speed:
        start = 'b'
        stats['first_attack'] = pokemon_b.name
    else:
        start = np.random.choice(['a', 'b'])
        if start == 'a':
            stats['first_attack'] = pokemon.name
        else:
            stats['first_attack'] = pokemon_b.name    

    while True:
        moves_exhausted = False
        
        if start == 'a':
            attacker = pokemon
            defender = pokemon_b
        else:
            attacker = pokemon_b
            defender = pokemon
        
        # starter attacks first
        stats['moves'] += 1
        move = choose_move(attacker)
        
        if move is not None:
            apply_move(attacker, defender, move)
        else:
            moves_exhausted = True
        
        if defender.current_hp <= 0:
            stats['winner'] = attacker.name
            break
        
        if start == 'a':
            attacker = pokemon_b
            defender = pokemon
        else:
            attacker = pokemon
            defender = pokemon_b
        
        # next pokemon attacks
        stats['moves'] += 1
        move = choose_move(attacker)
        
        if move is not None:
            apply_move(attacker, defender, move)
            moves_exhausted = False
        else:
            moves_exhausted = True
        
        if defender.current_hp <= 0:
            stats['winner'] = attacker.name
            break
        
        # handle case where all moves exhausted and no winner
        if moves_exhausted:
            stats['winner'] = None
            break
    
    return stats


def get_random_battle(base_level):
    random_pokemon_a = POKEMON_STATS.sample()['pokemon'].iloc[0]
    random_pokemon_b = POKEMON_STATS.sample()['pokemon'].iloc[0]
    level_a = base_level
    level_b = random.randint(max(0, level_a-2), min(level_a+2, 100))
    return (Pokemon(random_pokemon_a, level_a), Pokemon(random_pokemon_b, level_b))


def simulate_battle_many(opponents):
    """
    Parallelizable function that takes a tuple of pokemon names to battle.
    
    Ex:
    (pikachu, gastly)
    
    The two pokemon battle each other N number of times. The statistics
    are aggregated and returned.
    """
    stats = {
        'pokemon': [],
        'pokemonb': [],
        'avg_moves': [],
        'pokemon_wins': [],
        'pokemonb_wins': [],
        'ties': [],
    }
    
    pokemon = Pokemon(opponents[0])
    pokemon_b = Pokemon(opponents[1])
    print('{} vs {}'.format(pokemon.name.title(), pokemon_b.name.title()))
    
    battle_stats = {
        'a_wins': 0,
        'b_wins': 0,
        'ties': 0,
        'moves': [],
    }

    for _ in range(NUM_SIMULATIONS):
        result = simulate_battle(pokemon, pokemon_b)
        if result['winner'] == pokemon.name:
            battle_stats['a_wins'] += 1
        elif result['winner'] == pokemon_b.name:
            battle_stats['b_wins'] += 1
        else:
            battle_stats['ties'] += 1

        battle_stats['moves'].append(result['moves'])
    
    stats['pokemon'].append(pokemon.name)
    stats['pokemonb'].append(pokemon_b.name)
    stats['avg_moves'].append(np.array(battle_stats['moves']).mean())
    stats['pokemon_wins'].append(battle_stats['a_wins'])
    stats['pokemonb_wins'].append(battle_stats['b_wins'])
    stats['ties'].append(battle_stats['ties'])
    
    return pd.DataFrame(stats)


def main():
    # find pokemon that actually have damaging moves
    valid_pokemon = []
    for pokemon in POKEMON_AVAIL:
        p = Pokemon(pokemon)
        
        if p.has_moves:
            valid_pokemon.append(p.name)

    # construct pokemon matches as list of list where each sub-list contains
    # the two pokemon names that will battle
    battles = {}
    for i in valid_pokemon:
        for j in valid_pokemon:
            if i == j:
                continue
            
            opponents = [i, j]
            opponents.sort()
            battle_key = ','.join(opponents)
            battles[battle_key] = opponents
    
    matches = list(battles.values())
    
    # simulate the battles in parallel
    with Pool(cpu_count()) as pool:
        stats = pool.map(simulate_battle_many, matches)
    
    # output the results
    pd.concat(stats).to_csv(os.path.join(DATA_DIR, 'results', 'simulation_stats.csv'), index=False)


if __name__ == '__main__':
    main()