import logging
import os
from collections import defaultdict
from time import sleep

from web3.types import Wei
from web3_multi_provider import MultiProvider

from src import variables
from src.constants import SHARE_RATE_PRECISION_E27
from src.modules.accounting.typings import (
    ReportData,
    OracleReportData,
    PoolMetrics,
    AccountingProcessingState,
    LidoReportRebase,
    SharesRequestedToBurn,
)
from src.metrics.prometheus.accounting import (
    ACCOUNTING_IS_BUNKER,
    ACCOUNTING_CL_BALANCE_GWEI,
    ACCOUNTING_EL_REWARDS_VAULT_BALANCE_WEI,
    ACCOUNTING_WITHDRAWAL_VAULT_BALANCE_WEI
)
from src.metrics.prometheus.duration_meter import duration_meter
from src.services.validator_state import LidoValidatorStateService
from src.modules.submodules.consensus import ConsensusModule
from src.modules.submodules.oracle_module import BaseModule, ModuleExecuteDelay
from src.services.withdrawal import Withdrawal
from src.services.bunker import BunkerService
from src.typings import BlockStamp, Gwei, ReferenceBlockStamp
from src.utils.abi import named_tuple_to_dataclass
from src.utils.cache import global_lru_cache as lru_cache
from src.variables import ALLOW_REPORTING_IN_BUNKER_MODE
from src.web3py.typings import Web3
from src.web3py.extensions.lido_validators import StakingModule, NodeOperatorGlobalIndex, StakingModuleId


logger = logging.getLogger(__name__)

# 奖励库的地址
rewards_vault_address = os.environ['REWARDS_VAULT_ADDRESS']
if not Web3.is_checksum_address(rewards_vault_address):
    rewards_vault_address = Web3.to_checksum_address(rewards_vault_address)

# 数据块的块号
ORACLE_FROM_BLOCK = int(os.getenv('ORACLE_FROM_BLOCK', 0))

eth1_provider = os.environ['EXECUTION_CLIENT_URI']

provider = MultiProvider(eth1_provider.split(','))
w3 = Web3(provider)

if not w3.is_connected():
    logging.error('ETH node connection error!')
    exit(1)

class Accounting(BaseModule, ConsensusModule):
    """
    Accounting module updates the protocol TVL, distributes node-operator rewards, and processes user withdrawal requests.

    Report goes in tree phases:
        - Send report hash
        - Send report data (extra data hash inside)
            Contains information about lido states, last withdrawal request to finalize and exited validators count by module.
        - Send extra data
            Contains stuck and exited validators count by each node operator.
    """
    CONSENSUS_VERSION = 1
    CONTRACT_VERSION = 1

    def __init__(self, w3: Web3):
        self.report_contract = w3.lido_contracts.oracle
        super().__init__(w3)

        self.lido_validator_state_service = LidoValidatorStateService(self.w3)
        self.bunker_service = BunkerService(self.w3)

    def refresh_contracts(self):
        self.report_contract = self.w3.lido_contracts.oracle

    def execute_module(self, last_finalized_blockstamp: BlockStamp) -> ModuleExecuteDelay:
        report_blockstamp = self.get_blockstamp_for_report(last_finalized_blockstamp)

        if report_blockstamp:
            self.process_report(report_blockstamp)
            # Third phase of report. Specific for accounting.
            self.process_extra_data(report_blockstamp)
            return ModuleExecuteDelay.NEXT_SLOT

        return ModuleExecuteDelay.NEXT_FINALIZED_EPOCH

    def process_extra_data(self, blockstamp: ReferenceBlockStamp):
        latest_blockstamp = self._get_latest_blockstamp()
        if not self.can_submit_extra_data(latest_blockstamp):
            logger.info({'msg': 'Extra data can not be submitted.'})
            return

        chain_config = self.get_chain_config(blockstamp)
        slots_to_sleep = self._get_slot_delay_before_data_submit(blockstamp)
        seconds_to_sleep = slots_to_sleep * chain_config.seconds_per_slot
        logger.info({'msg': f'Sleep for {seconds_to_sleep} seconds before sending extra data.'})
        sleep(seconds_to_sleep)

        latest_blockstamp = self._get_latest_blockstamp()
        if not self.can_submit_extra_data(latest_blockstamp):
            logger.info({'msg': 'Extra data can not be submitted.'})
            return

        self._submit_extra_data(blockstamp)

    def _submit_extra_data(self, blockstamp: ReferenceBlockStamp) -> None:
        extra_data = self.lido_validator_state_service.get_extra_data(blockstamp, self.get_chain_config(blockstamp))

        if extra_data.extra_data:
            tx = self.report_contract.functions.submitReportExtraDataList(extra_data.extra_data)
        else:
            tx = self.report_contract.functions.submitReportExtraDataEmpty()

        self.w3.transaction.check_and_send_transaction(tx, variables.ACCOUNT)

    @lru_cache(maxsize=1)
    @duration_meter()
    def build_report(self, blockstamp: ReferenceBlockStamp) -> tuple:
        report_data = self._calculate_report(blockstamp)
        logger.info({'msg': 'Calculate report for accounting module.', 'value': report_data})
        return report_data.as_tuple()

    def is_main_data_submitted(self, blockstamp: BlockStamp) -> bool:
        # Consensus module: if contract got report data (second phase)
        processing_state = self._get_processing_state(blockstamp)
        logger.debug({'msg': 'Check if main data was submitted.', 'value': processing_state.main_data_submitted})
        return processing_state.main_data_submitted

    def can_submit_extra_data(self, blockstamp: BlockStamp) -> bool:
        """Check if Oracle can submit extra data. Can only be submitted after second phase."""
        processing_state = self._get_processing_state(blockstamp)
        return processing_state.main_data_submitted and not processing_state.extra_data_submitted

    def is_contract_reportable(self, blockstamp: BlockStamp) -> bool:
        # Consensus module: if contract can accept the report (in any phase)
        is_reportable = not self.is_main_data_submitted(blockstamp) or self.can_submit_extra_data(blockstamp)
        logger.info({'msg': 'Check if contract could accept report.', 'value': is_reportable})
        return is_reportable

    def is_reporting_allowed(self, blockstamp: ReferenceBlockStamp) -> bool:
        if not self._is_bunker(blockstamp):
            return True

        logger.warning({'msg': '!' * 50})
        logger.warning({'msg': f'Bunker mode is active. {ALLOW_REPORTING_IN_BUNKER_MODE=}'})
        logger.warning({'msg': '!' * 50})
        return ALLOW_REPORTING_IN_BUNKER_MODE

    @lru_cache(maxsize=1)
    def _get_processing_state(self, blockstamp: BlockStamp) -> AccountingProcessingState:
        ps = named_tuple_to_dataclass(
            self.report_contract.functions.getProcessingState().call(block_identifier=blockstamp.block_hash),
            AccountingProcessingState,
        )
        logger.info({'msg': 'Fetch processing state.', 'value': ps})
        return ps

    # ---------------------------------------- Build report ----------------------------------------
    def _calculate_report(self, blockstamp: ReferenceBlockStamp) -> ReportData:
        beacon_spec = self.w3.lido_contracts.oracle.functions.getBeaconSpec().call()
        logger.info({'msg': 'Fetch beacon spec.', 'value': beacon_spec})
        # 获取之前上报的数据
        prev_metrics = self.get_previous_metrics(beacon_spec,  ORACLE_FROM_BLOCK)
        logger.info({'msg': 'Fetch prev metrics.', 'value': prev_metrics})
        logging.info(f'Previously reported epoch: {prev_metrics.epoch}')
        logging.info(f'Previously reported bufferedBalance: {prev_metrics.bufferedBalance}')

        validators_count, cl_balance = self._get_consensus_lido_state(blockstamp)

        staking_module_ids_list, exit_validators_count_list = self._get_newly_exited_validators_by_modules(blockstamp)

        extra_data = self.lido_validator_state_service.get_extra_data(blockstamp, self.get_chain_config(blockstamp))
        finalization_share_rate, finalization_batches = self._get_finalization_data(blockstamp)

        report_data = ReportData(
            consensus_version=self.CONSENSUS_VERSION,
            ref_slot=blockstamp.ref_slot,
            validators_count=validators_count,
            cl_balance_gwei=cl_balance,
            staking_module_id_with_exited_validators=staking_module_ids_list,
            count_exited_validators_by_staking_module=exit_validators_count_list,
            withdrawal_vault_balance=self.w3.lido_contracts.get_withdrawal_balance(blockstamp),
            el_rewards_vault_balance=self.w3.lido_contracts.get_el_vault_balance(blockstamp),
            shares_requested_to_burn=self.get_shares_to_burn(blockstamp),
            withdrawal_finalization_batches=finalization_batches,
            finalization_share_rate=finalization_share_rate,
            is_bunker=self._is_bunker(blockstamp),
            extra_data_format=extra_data.format,
            extra_data_hash=extra_data.data_hash,
            extra_data_items_count=extra_data.items_count,
        )

        ACCOUNTING_IS_BUNKER.set(report_data.is_bunker)
        ACCOUNTING_CL_BALANCE_GWEI.set(report_data.cl_balance_gwei)
        ACCOUNTING_EL_REWARDS_VAULT_BALANCE_WEI.set(report_data.el_rewards_vault_balance)
        ACCOUNTING_WITHDRAWAL_VAULT_BALANCE_WEI.set(report_data.withdrawal_vault_balance)

        return report_data

    def get_previous_metrics(self, beacon_spec,  from_block=0) -> PoolMetrics:
        logging.info('Getting previously reported numbers (will be fetched from events)...')
        genesis_time = beacon_spec[3]
        result = PoolMetrics()
        # 通过abi调用pool合约
        result.preDepositValidators, result.depositedValidators, result.beaconValidators, result.beaconBalance = self.w3.lido_contracts.pool.functions.getBeaconStat().call()
        # 缓冲余额(将缓冲的 eth 存入质押合约并将分块存款分配给节点运营商)
        result.bufferedBalance = self.w3.lido_contracts.pool.functions.getBufferedEther().call()
        # Calculate the earliest block to limit scanning depth 计算最早的块以限制扫描深度
        # 每个 ETH1 块的秒数 正常12s 出一个块 设置14s为了遍历
        SECONDS_PER_ETH1_BLOCK = 14
        latest_block = w3.eth.get_block('latest')
        from_block = max(from_block, int((latest_block['timestamp'] - genesis_time) / SECONDS_PER_ETH1_BLOCK))

        latest_num = latest_block['number']
        logging.info(f'DawnPool from_block : {from_block}, latest_num : {latest_num}')
        # Try to fetch and parse last 'Completed' event from the contract. 遍历从合约中获取并解析最后一个“已完成”事件。
        step = 1000
        for end in range(latest_block['number'], from_block, -step):
            start = max(end - step + 1, from_block)
            # 调用了 getLogs 方法来读取区块链上合约中 Completed 事件在指定区块高度范围内的日志,日志会被存储在 events 变量中
            events = self.w3.lido_contracts.oracle.events.Completed.get_logs(fromBlock=start, toBlock=end)
            # 判断 events 是否为空 如果存在符合条件的事件日志，获取最后一个事件，即 events[-1]，并从中提取出相应的信息
            if events:
                logging.info(f'DawnPool event is : {events}')
                event = events[-1]
                result.epoch = event['args']['epochId']
                result.blockNumber = event.blockNumber
                break

        #  查询奖励库的地址(配置在环境变量里)对应账户在指定区块高度时的余额
        result.rewardsVaultBalance = w3.eth.get_balance(
            w3.to_checksum_address(rewards_vault_address.replace('0x010000000000000000000000', '0x')),
            block_identifier=result.blockNumber
        )
        logging.info(f'DawnPool result : {result}')
        # If the epoch has been assigned from the last event (not the first run) 如果纪元是从最后一个事件（不是第一次运行）分配的
        if result.epoch:
            result.timestamp = self.get_timestamp_by_epoch(beacon_spec, result.epoch)
        else:
            # If it's the first run, we set timestamp to genesis time 如果是第一次运行，我们将时间戳设置为创世时间
            result.timestamp = genesis_time
        return result

    def get_timestamp_by_epoch(self,beacon_spec, epoch_id):
        """Required to calculate time-bound values such as APR"""
        slots_per_epoch = beacon_spec[1]
        seconds_per_slot = beacon_spec[2]
        genesis_time = beacon_spec[3]
        return genesis_time + slots_per_epoch * seconds_per_slot * epoch_id

    def _get_newly_exited_validators_by_modules(
        self,
        blockstamp: ReferenceBlockStamp,
    ) -> tuple[list[StakingModuleId], list[int]]:
        """
        Calculate exited validators count in all modules.
        Exclude modules without changes from the report.
        """
        staking_modules = self.w3.lido_validators.get_staking_modules(blockstamp)
        exited_validators = self.lido_validator_state_service.get_exited_lido_validators(blockstamp)

        return self.get_updated_modules_stats(staking_modules, exited_validators)

    @staticmethod
    def get_updated_modules_stats(
        staking_modules: list[StakingModule],
        exited_validators_by_no: dict[NodeOperatorGlobalIndex, int],
    ) -> tuple[list[StakingModuleId], list[int]]:
        """Returns exited validators count by node operators that should be updated."""
        module_stats: dict[StakingModuleId, int] = defaultdict(int)

        for (module_id, _), validators_exited_count in exited_validators_by_no.items():
            module_stats[module_id] += validators_exited_count

        for module in staking_modules:
            if module_stats[module.id] == module.exited_validators_count:
                del module_stats[module.id]

        return list(module_stats.keys()), list(module_stats.values())

    @lru_cache(maxsize=1)
    def _get_consensus_lido_state(self, blockstamp: ReferenceBlockStamp) -> tuple[int, Gwei]:
        lido_validators = self.w3.lido_validators.get_lido_validators(blockstamp)

        count = len(lido_validators)
        total_balance = Gwei(sum(int(validator.balance) for validator in lido_validators))

        logger.info({'msg': 'Calculate consensus lido state.', 'value': (count, total_balance)})
        return count, total_balance

    def _get_finalization_data(self, blockstamp: ReferenceBlockStamp) -> tuple[int, list[int]]:
        simulation = self.simulate_full_rebase(blockstamp)
        chain_config = self.get_chain_config(blockstamp)
        frame_config = self.get_frame_config(blockstamp)
        is_bunker = self._is_bunker(blockstamp)

        share_rate = simulation.post_total_pooled_ether * SHARE_RATE_PRECISION_E27 // simulation.post_total_shares
        logger.info({'msg': 'Calculate shares rate.', 'value': share_rate})

        withdrawal_service = Withdrawal(self.w3, blockstamp, chain_config, frame_config)
        batches = withdrawal_service.get_finalization_batches(
            is_bunker,
            share_rate,
            simulation.withdrawals,
            simulation.el_reward,
        )

        logger.info({'msg': 'Calculate last withdrawal id to finalize.', 'value': batches})

        return share_rate, batches

    @lru_cache(maxsize=1)
    def simulate_cl_rebase(self, blockstamp: ReferenceBlockStamp) -> LidoReportRebase:
        """
        Simulate rebase excluding any execution rewards.
        This used to check worst scenarios in bunker service.
        """
        return self.simulate_rebase_after_report(blockstamp, el_rewards=Wei(0))

    def simulate_full_rebase(self, blockstamp: ReferenceBlockStamp) -> LidoReportRebase:
        el_rewards = self.w3.lido_contracts.get_el_vault_balance(blockstamp)
        return self.simulate_rebase_after_report(blockstamp, el_rewards=el_rewards)

    def simulate_rebase_after_report(
        self,
        blockstamp: ReferenceBlockStamp,
        el_rewards: Wei,
    ) -> LidoReportRebase:
        """
        To calculate how much withdrawal request protocol can finalize - needs finalization share rate after this report
        """
        validators_count, cl_balance = self._get_consensus_lido_state(blockstamp)

        chain_conf = self.get_chain_config(blockstamp)

        simulated_tx = self.w3.lido_contracts.lido.functions.handleOracleReport(
            # We use block timestamp, instead of slot timestamp,
            # because missed slot will break simulation contract logics
            # Details: https://github.com/lidofinance/lido-oracle/issues/291
            blockstamp.block_timestamp,  # _reportTimestamp
            self._get_slots_elapsed_from_last_report(blockstamp) * chain_conf.seconds_per_slot,  # _timeElapsed
            # CL values
            validators_count,  # _clValidators
            Web3.to_wei(cl_balance, 'gwei'),  # _clBalance
            # EL values
            self.w3.lido_contracts.get_withdrawal_balance(blockstamp),  # _withdrawalVaultBalance
            el_rewards,  # _elRewardsVaultBalance
            self.get_shares_to_burn(blockstamp),  # _sharesRequestedToBurn
            # Decision about withdrawals processing
            [],  # _lastFinalizableRequestId
            0,  # _simulatedShareRate
        )

        logger.info({'msg': 'Simulate lido rebase for report.', 'value': simulated_tx.args})

        result = simulated_tx.call(
            transaction={'from': self.w3.lido_contracts.oracle.address},
            block_identifier=blockstamp.block_hash,
        )

        logger.info({'msg': 'Fetch simulated lido rebase for report.', 'value': result})

        return LidoReportRebase(*result)

    @lru_cache(maxsize=1)
    def get_shares_to_burn(self, blockstamp: BlockStamp) -> int:
        shares_data = named_tuple_to_dataclass(
            self.w3.lido_contracts.burner.functions.getSharesRequestedToBurn().call(
                block_identifier=blockstamp.block_hash,
            ),
            SharesRequestedToBurn,
        )

        return shares_data.cover_shares + shares_data.non_cover_shares

    def _get_slots_elapsed_from_last_report(self, blockstamp: ReferenceBlockStamp):
        chain_conf = self.get_chain_config(blockstamp)
        frame_config = self.get_frame_config(blockstamp)

        last_ref_slot = self.w3.lido_contracts.get_accounting_last_processing_ref_slot(blockstamp)

        if last_ref_slot:
            slots_elapsed = blockstamp.ref_slot - last_ref_slot
        else:
            slots_elapsed = blockstamp.ref_slot - frame_config.initial_epoch * chain_conf.slots_per_epoch

        return slots_elapsed

    @lru_cache(maxsize=1)
    def _is_bunker(self, blockstamp: ReferenceBlockStamp) -> bool:
        frame_config = self.get_frame_config(blockstamp)
        chain_config = self.get_chain_config(blockstamp)
        cl_rebase_report = self.simulate_cl_rebase(blockstamp)

        bunker_mode = self.bunker_service.is_bunker_mode(
            blockstamp,
            frame_config,
            chain_config,
            cl_rebase_report,
        )
        logger.info({'msg': 'Calculate bunker mode.', 'value': bunker_mode})
        return bunker_mode
