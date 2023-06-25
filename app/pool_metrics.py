class PoolMetrics:
    DEPOSIT_SIZE = int(32 * 1e18)
    MAX_APR = 0.15
    MIN_APR = 0.01
    epoch = 0
    finalized_epoch_beacon = 0
    beaconBalance = 0
    beaconValidators = 0
    timestamp = 0
    blockNumber = 0
    bufferedBalance = 0
    preDepositValidators = 0
    depositedValidators = 0
    activeValidatorBalance = 0
    withdrawalVaultBalance = 0
    exitedValidatorsCount = 0
    burnedPethAmount = 0
    lastRequestIdToBeFulfilled = 0
    ethAmountToLock = 0
    validatorsKeysNumber = None

    def getTotalPooledEther(self):
        return self.bufferedBalance + self.beaconBalance + self.getTransientBalance()

    def getTransientValidators(self):

        self.depositedValidators + self.preDepositValidators >= self.beaconValidators
        # assert self.depositedValidators >= self.beaconValidators
        return self.depositedValidators - self.beaconValidators

    def getTransientBalance(self):
        return self.getTransientValidators() * self.DEPOSIT_SIZE
