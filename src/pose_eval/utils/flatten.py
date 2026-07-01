# flatten nested dictionary
from pprint import pprint


def flatten(d):
    pprint(d)
    flat_dict = {}
    for condition, result in d.items():
        for key, value in result.items():
            flat_dict[f'{condition}_{key}'] = value
    return flat_dict

if __name__ == '__main__':
    d = { 'a': {'loss': 3, 'x': 2}, 'b': {'loss': 12, 'x': 7}}
    flat_d = flatten(d)
    print(f'flattened once: {flat_d}')
    flat_flat_d = flatten(flat_d)
    print(f'flattened twice: {flat_flat_d}')

