import random
import subprocess
import json

# generates random value for each solidity data type
def rand_value(tp):
    if tp == 'int' or tp == 'int256':
        return random.randint(-2 ** 255, 2 ** 255 - 1)
    if tp == 'int128':
        return random.randint(-2 ** 127, 2 ** 127 -1)

    if tp == 'uint' or tp == 'uint256':
        return random.randint(0, 2 ** 256 - 1) 
    if tp == 'uint128': 
        return random.randint(0, 2 ** 128 - 1)
    if tp == 'uint8':
        return random.randint(0, 2 ** 8 - 1)
    # TODO: same for all solidity types
    raise f"Not valid data type: {tp}"


class method:
    def __init__(self, decl: str):
        args_str = decl[decl.find('(') + 1:decl.find(')')]
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


# runs wasm contract in wasmtime with string input and returns dict: {"status" : "<status>", "output": "<output>"}
def run_in_wasmtime(contract: str, inpt: str) -> json:
    name = inpt[:inpt.find(' ')]
    args = inpt[inpt.find(' ') + 4:-1]
    args = args[1:-1]
    res = subprocess.run(["wasmtime", "--allow-unknown-exports", contract, "--invoke", name, "--", args, "/dev/null"], capture_output=True)
    output = bytes.fromhex(res.stderr.decode().split('\n')[-2][9:]).decode()
    json_out = json.loads(output)
    return json_out




if __name__ == "__main__":
    # dump_test_inputs("test/calc.sol", "test_inputs/calc.txt", 100)
    with open("test_inputs/calc.txt", "r") as inpts:
        testline = inpts.readlines()[0]
        run_in_wasmtime("calc.wasm", testline)
