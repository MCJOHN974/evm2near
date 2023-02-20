const contracts = [
  artifacts.require("Calc"),
  artifacts.require("Bench"),
  artifacts.require("Collatz"),
  artifacts.require("Const"),
  artifacts.require("Echo")
];

module.exports = function (deployer) {
  for (const contract of contracts) {
    deployer.deploy(contract);
  }
};
