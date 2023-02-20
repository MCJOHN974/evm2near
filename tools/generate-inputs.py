import random
import subprocess
import json
import os


# generates random value for each solidity data type
def rand_value(tp):
    if tp == 'int' or tp == 'int256':
        return random.randint(-2 ** 255, 2 ** 255 - 1)
    if tp == 'int128':
        return random.randint(-2 ** 127, 2 ** 127 -1)
    if tp == 'int64':
        return random.randint(-2 ** 63, 2 ** 63 -1)
    if tp == 'int32':
        return random.randint(-2 ** 31, 2 ** 31 -1)
    if tp == 'int16':
        return random.randint(-2 ** 15, 2 ** 15 -1)
    if tp == 'int8':
        return random.randint(-2 ** 7, 2 ** 7 -1)

    if tp == 'uint' or tp == 'uint256':
        return random.randint(0, 2 ** 256 - 1) 
    if tp == 'uint128': 
        return random.randint(0, 2 ** 128 - 1)
    if tp == 'uint64': 
        return random.randint(0, 2 ** 64 - 1)
    if tp == 'uint32': 
        return random.randint(0, 2 ** 32 - 1)
    if tp == 'uint16': 
        return random.randint(0, 2 ** 16 - 1)
    if tp == 'uint8':
        return random.randint(0, 2 ** 8 - 1)
    # TODO: same for all solidity types
    raise f"Not valid data type: {tp}"


class method:
    def __init__(self, decl: str):
        args_str = decl[decl.find('(') + 1:decl.find(')')]
        if len(args_str) == 0:
            self.args = dict()
        else:
            self.args = dict(map(lambda x: (x.strip().split(' ')[1], x.strip().split(' ')[0]), args_str.split(',')))
        self.out = decl[decl.rfind('(') + 1:decl.rfind(')')].strip()
        self.name = decl[decl.find('function') + 8:decl.find('(')].strip()


    def random_input(self):
        return dict(map(lambda name: (name, rand_value(self.args[name])), self.args.keys()))


# returns list of methods of contract
def signatures(contract: str) -> list[method]:
    with open(contract, 'r') as code:
        lines = code.readlines()
        declarations = list(filter(lambda x: 'function' in x and 'public' in x, lines))
        return list(map(lambda x: method(x), declarations))


def dump_test_inputs(contract: str, file: str, number_of_tests_for_method: int):
    methods = signatures(contract)
    lines = []
    for method in methods:
        for _ in range(number_of_tests_for_method):
            lines.append(str(method.name) + ' -- \'' + str(method.random_input()).replace("'", "\"") + '\'\n')
    with open(file, 'w') as test_file:
        test_file.writelines(lines)


if __name__ == "__main__":
    for contract in os.listdir('test'):
        contract = contract[:-4]
        dump_test_inputs(f"etherium/contracts/{contract}.sol", f"test_inputs/{contract}.txt", 100)
