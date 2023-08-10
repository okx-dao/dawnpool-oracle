import json
import logging
import os
from time import sleep

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import BadFunctionCallOutput
from web3.module import Module
from web3.types import Wei

from src import variables
from src.metrics.prometheus.business import FRAME_PREV_REPORT_REF_SLOT, FRAME_PREV_REPORT_REF_EPOCH
from src.typings import BlockStamp, SlotNumber
from src.utils.cache import global_lru_cache as lru_cache

logger = logging.getLogger()

pool_address = os.environ['POOL_CONTRACT']
if not Web3.is_checksum_address(pool_address):
    pool_address = Web3.to_checksum_address(pool_address)

oracle_address = os.environ['ORACLE_CONTRACT']
if not Web3.is_checksum_address(oracle_address):
    oracle_address = Web3.to_checksum_address(oracle_address)

node_manager_address = os.environ['NODE_MANAGER_CONTRACT']
if not Web3.is_checksum_address(node_manager_address):
    node_manager_address = Web3.to_checksum_address(node_manager_address)

withdraw_address = os.environ['WITHDRAW_CONTRACT']
if not Web3.is_checksum_address(withdraw_address):
    withdraw_address = Web3.to_checksum_address(withdraw_address)

validators_exit_bus_address = os.environ['VALIDATORS_EXIT_BUS_ORACLE_CONTRACT']
if not Web3.is_checksum_address(validators_exit_bus_address):
    validators_exit_bus_address = Web3.to_checksum_address(validators_exit_bus_address)

# 奖励库的地址
rewards_vault_address = os.environ['REWARDS_VAULT_ADDRESS']
if not Web3.is_checksum_address(rewards_vault_address):
    rewards_vault_address = Web3.to_checksum_address(rewards_vault_address)

burner_address = os.environ['BURNER_CONTRACT']
if not Web3.is_checksum_address(burner_address):
    burner_address = Web3.to_checksum_address(burner_address)

class LidoContracts(Module):
    lido_locator: Contract
    lido: Contract
    accounting_oracle: Contract
    staking_router: Contract
    validators_exit_bus_oracle: Contract
    withdrawal_queue_nft: Contract
    oracle_report_sanity_checker: Contract
    oracle_daemon_config: Contract
    burner: Contract
    pool: Contract
    oracle: Contract
    withdraw: Contract
    registry: Contract


    def __init__(self, w3: Web3):
        super().__init__(w3)
        self._load_contracts()

    def __setattr__(self, key, value):
        current_value = getattr(self, key, None)
        if isinstance(current_value, Contract) and isinstance(value, Contract):
            if value.address != current_value.address:
                logger.info({'msg': f'Contract {key} has been changed to {value.address}'})
        super().__setattr__(key, value)

    def has_contract_address_changed(self) -> bool:
        addresses = [contract.address for contract in self.__dict__.values() if isinstance(contract, Contract)]
        self._load_contracts()
        new_addresses = [contract.address for contract in self.__dict__.values() if isinstance(contract, Contract)]
        return addresses != new_addresses

    # def _check_contracts(self):
    #     """This is startup check that checks that contract are deployed and has valid implementation"""
    #     try:
    #         self.accounting_oracle.functions.getContractVersion().call()
    #         self.validators_exit_bus_oracle.functions.getContractVersion().call()
    #     except BadFunctionCallOutput:
    #         logger.info({
    #             'msg': 'getContractVersion method from accounting_oracle and validators_exit_bus_oracle '
    #                    'doesn\'t return any data. Probably addresses from Lido Locator refer to the wrong '
    #                    'implementation or contracts don\'t exist. Sleep for 1 minute.'
    #         })
    #         sleep(60)
    #         self._load_contracts()
    #     else:
    #         return

    def _check_contracts(self):
        self._load_contracts()

    def _load_contracts(self):
        # Contract that stores all lido contract addresses
        # self.lido_locator = self.w3.eth.contract(
        #     address=variables.LIDO_LOCATOR_ADDRESS,
        #     abi=self.load_abi('LidoLocator'),
        #     decode_tuples=True,
        # )

        # self.lido = self.w3.eth.contract(
        #     address=self.lido_locator.functions.lido().call(),
        #     abi=self.load_abi('Lido'),
        #     decode_tuples=True,
        # )

        # self.accounting_oracle = self.w3.eth.contract(
        #     address=self.lido_locator.functions.accountingOracle().call(),
        #     abi=self.load_abi('AccountingOracle'),
        #     decode_tuples=True,
        # )

        # self.staking_router = self.w3.eth.contract(
        #     address=self.lido_locator.functions.stakingRouter().call(),
        #     abi=self.load_abi('StakingRouter'),
        #     decode_tuples=True,
        # )

        # self.validators_exit_bus_oracle = self.w3.eth.contract(
        #     address=self.lido_locator.functions.validatorsExitBusOracle().call(),
        #     abi=self.load_abi('ValidatorsExitBusOracle'),
        #     decode_tuples=True,
        # )

        self.validators_exit_bus_oracle = self.w3.eth.contract(
            address=validators_exit_bus_address,
            abi=self.load_abi('ValidatorsExitBusOracle'),
            decode_tuples=True,
        )

        # self.withdrawal_queue_nft = self.w3.eth.contract(
        #     address=self.lido_locator.functions.withdrawalQueue().call(),
        #     abi=self.load_abi('WithdrawalQueueERC721'),
        #     decode_tuples=True,
        # )

        # self.oracle_report_sanity_checker = self.w3.eth.contract(
        #     address=self.lido_locator.functions.oracleReportSanityChecker().call(),
        #     abi=self.load_abi('OracleReportSanityChecker'),
        #     decode_tuples=True,
        # )
        #
        # self.oracle_daemon_config = self.w3.eth.contract(
        #     address=self.lido_locator.functions.oracleDaemonConfig().call(),
        #     abi=self.load_abi('OracleDaemonConfig'),
        #     decode_tuples=True,
        # )

        self.burner = self.w3.eth.contract(
            address=burner_address,
            abi=self.load_abi('Burner'),
            decode_tuples=True,
        )

        self.pool = self.w3.eth.contract(
            address=pool_address,
            abi=self.load_abi('DawnDeposit'),
            decode_tuples=True,
        )

        self.oracle = self.w3.eth.contract(
            address=oracle_address,
            abi=self.load_abi('DawnPoolOracle'),
            decode_tuples=True,
        )

        self.registry = self.w3.eth.contract(
            address=node_manager_address,
            abi=self.load_abi('DepositNodeManager'),
            decode_tuples=True,
        )

        self.withdraw = self.w3.eth.contract(
            address=withdraw_address,
            abi=self.load_abi('DawnWithdraw'),
            decode_tuples=True,
        )

        # self._check_contracts()

    @staticmethod
    def load_abi(abi_name: str, abi_path: str = './assets/'):
        with open(f'{abi_path}{abi_name}.json') as f:
            return json.load(f)

    # --- Contract methods ---
    @lru_cache(maxsize=1)
    def get_withdrawal_balance(self, blockstamp: BlockStamp) -> Wei:
        return self.get_withdrawal_balance_no_cache(blockstamp)

    def get_withdrawal_balance_no_cache(self, blockstamp: BlockStamp) -> Wei:
        return Wei(self.w3.eth.get_balance(
            self.lido_locator.functions.withdrawalVault().call(
                block_identifier=blockstamp.block_hash
            ),
            block_identifier=blockstamp.block_hash,
        ))

    @lru_cache(maxsize=1)
    def get_el_vault_balance(self, blockstamp: BlockStamp) -> Wei:
        #  downstorage  RewardsVault地址的余额
        return Wei(self.w3.eth.get_balance(
            self.lido_locator.functions.elRewardsVault().call(
                block_identifier=blockstamp.block_hash
            ),
            block_identifier=blockstamp.block_hash,
        ))

    @lru_cache(maxsize=1)
    def get_accounting_last_processing_ref_slot(self, blockstamp: BlockStamp) -> SlotNumber:
        result = self.accounting_oracle.functions.getLastProcessingRefSlot().call(block_identifier=blockstamp.block_hash)
        logger.info({'msg': f'Accounting last processing ref slot {result}'})
        FRAME_PREV_REPORT_REF_SLOT.set(result)
        return result

    @lru_cache(maxsize=1)
    def get_accounting_last_processing_ref_epoch(self, blockstamp: BlockStamp) -> SlotNumber:
        result = self.w3.lido_contracts.oracle.functions.getLastProcessingRefSlot().call(block_identifier=blockstamp.block_hash)
        logger.info({'msg': f'Accounting last processing ref epoch {result}'})
        FRAME_PREV_REPORT_REF_SLOT.set(result)
        return result

    def get_ejector_last_processing_ref_epoch(self, blockstamp: BlockStamp) -> SlotNumber:

        result = self.w3.lido_contracts.validators_exit_bus_oracle.functions.getLastProcessingRefSlot().call(
            block_identifier=blockstamp.block_hash
        )
        logger.info({'msg': f'Ejector last processing ref epoch {result}'})
        FRAME_PREV_REPORT_REF_EPOCH.set(result)
        return result
