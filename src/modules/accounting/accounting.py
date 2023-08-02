import binascii
import decimal
import logging
import os
import sys
from collections import defaultdict
from time import sleep

from web3.exceptions import ContractLogicError
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
from src.modules.submodules.typings import ChainConfig
from src.providers.consensus.typings import ValidatorStatus
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

        if not report_blockstamp:
            return ModuleExecuteDelay.NEXT_FINALIZED_EPOCH
        # if report_blockstamp:
        #     self.process_report(report_blockstamp)
        #     # Third phase of report. Specific for accounting.
        #     self.process_extra_data(report_blockstamp)
        #     return ModuleExecuteDelay.NEXT_SLOT

        self.process_report(report_blockstamp)
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
        prev_metrics = self.get_previous_metrics(beacon_spec, ORACLE_FROM_BLOCK)
        if prev_metrics:
            logging.info(f'Previously reported epoch: {prev_metrics.epoch}')
            logging.info(
                f'Previously reported beaconBalance: {prev_metrics.beaconBalance} wei or {prev_metrics.beaconBalance / 1e18} ETH'
            )
            logging.info(
                f'Previously reported bufferedBalance: {prev_metrics.bufferedBalance} wei or {prev_metrics.bufferedBalance / 1e18} ETH'
            )
            logging.info(f'Previous validator metrics: depositedValidators:{prev_metrics.depositedValidators}')
            logging.info(f'Previous validator metrics: beaconValidators:{prev_metrics.beaconValidators}')
            logging.info(f'Previous validator metrics: rewardsVaultBalance:{prev_metrics.rewardsVaultBalance}')
            logging.info(
                f'Timestamp of previous report:  {prev_metrics.timestamp}'
            )

        current_metrics = self.get_light_current_metrics(beacon_spec)
        logging.info(
            f'Currently Metrics epoch: {current_metrics.epoch} Prev Metrics epoch {prev_metrics.epoch} '
        )
        # 已经提交过
        if current_metrics.epoch <= prev_metrics.epoch:
            logging.info(f'Currently reportable epoch {current_metrics.epoch} has already been reported. Skipping it.')
            # 已经上报过,退出进程
            # sys.exit()

        # Get full metrics using polling (get keys from registry, get balances from beacon)
        current_metrics = self.get_full_current_metrics(blockstamp, beacon_spec, current_metrics)

        # 对比
        self.compare_pool_metrics(prev_metrics, current_metrics)

        report_data = OracleReportData(
            epoch_id=current_metrics.epoch,
            beacon_balance=current_metrics.activeValidatorBalance,
            beacon_validators=current_metrics.beaconValidators,
            rewards_vault_balance=current_metrics.rewardsVaultBalance,
            exited_validators=current_metrics.exitedValidatorsCount,
            burned_peth_amount=current_metrics.burnedPethAmount,
            last_request_id_to_be_fulfilled=current_metrics.lastRequestIdToBeFulfilled,
            eth_amount_to_lock=current_metrics.ethAmountToLock
        )

        return report_data

    def get_previous_metrics(self, beacon_spec, from_block=0) -> PoolMetrics:
        logging.info('Getting previously reported numbers (will be fetched from events)...')
        genesis_time = beacon_spec[3]
        result = PoolMetrics()
        # 通过abi调用pool合约
        result.preDepositValidators, result.depositedValidators, result.beaconValidators, result.beaconBalance = self.w3.lido_contracts.pool.functions.getBeaconStat().call()
        # 缓冲余额(将缓冲的 eth 存入质押合约并将分块存款分配给节点运营商)
        result.bufferedBalance = self.w3.lido_contracts.pool.functions.getBufferedEther().call()
        # Calculate the earliest block to limit scanning depth 计算最早的块以限制扫描深度
        # 每个 ETH1 块的秒数 正常12s 出一个块
        SECONDS_PER_ETH1_BLOCK = 12
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

    def get_light_current_metrics(self, beacon_spec):
        """Fetch current frame, buffered balance and epoch"""
        # 每帧的epoch数
        epochs_per_frame = beacon_spec[0]
        partial_metrics = PoolMetrics()
        partial_metrics.blockNumber = w3.eth.get_block('latest')[
            'number']  # Get the epoch that is finalized and reportable
        # 当前帧信息的数组  通过abi合约调用查询
        current_frame = self.w3.lido_contracts.oracle.functions.getCurrentFrame().call()
        # 当前帧所在的epoch，作为潜在的报告epoch
        potentially_reportable_epoch = current_frame[0]
        logging.info(f'Potentially reportable epoch: {potentially_reportable_epoch} (from ETH1 contract)')
        # 获得最终确定的纪元
        finalized_epoch_beacon = self.w3.cc.get_finalized_epoch()
        # For Web3 client
        # finalized_epoch_beacon = int(beacon.get_finality_checkpoint()['data']['finalized']['epoch'])
        logging.info(f'Last finalized epoch: {finalized_epoch_beacon} (from Beacon)')
        # //是向下取整  第二个通过计算得出的实际报告时代 计算信标链中已经最终化的时代数 finalized_epoch_beacon 所在的当前帧
        partial_metrics.epoch = min(
            potentially_reportable_epoch, (finalized_epoch_beacon // epochs_per_frame) * epochs_per_frame
        )
        partial_metrics.timestamp = self.get_timestamp_by_epoch(beacon_spec, partial_metrics.epoch)
        partial_metrics.depositedValidators = self.w3.lido_contracts.pool.functions.getBeaconStat().call()[0]
        partial_metrics.bufferedBalance = self.w3.lido_contracts.pool.functions.getBufferedEther().call()
        return partial_metrics

    def get_full_current_metrics(self, blockstamp: ReferenceBlockStamp,
                                 beacon_spec: ChainConfig,
                                 partial_metrics: PoolMetrics) -> PoolMetrics:
        """The oracle fetches all the required states from ETH1 and ETH2 (validator balances)"""
        slots_per_epoch = beacon_spec[1]
        logging.info(f'Reportable slots_per_epoch: {slots_per_epoch} ,partial_metrics.epoch: {partial_metrics.epoch}')
        slot = partial_metrics.epoch * slots_per_epoch
        logging.info(f'Reportable state, epoch:{partial_metrics.epoch} slot:{slot}')

        block_number = self.w3.cc.get_block_by_beacon_slot(slot, beacon_spec[0], beacon_spec[1])

        logging.info(f'Validator block_number: {block_number}')

        #  获取注册的验证者的key 通过abi获取
        validators_keys = self.w3.lido_contracts.registry.functions.getNodeValidators(0, 0).call()[1]

        hex_validators_keys = tuple('0x' + binascii.hexlify(pk).decode('ascii') for pk in validators_keys)

        logging.info({'msg': 'Fetch dawn pool status keys.', 'value': hex_validators_keys})
        logging.info(f'Total validator keys in registry: {len(validators_keys)}')
        full_metrics = partial_metrics

        validators = self.w3.cc.get_pub_key_validators(blockstamp, hex_validators_keys)
        logger.info({'msg': 'Fetch dawn pool validators.', 'value': validators})

        validators_count = 0
        total_balance = 0
        active_validators_balance = 0
        exited_validators_count = 0

        # todo
        for validator in validators:
            logging.info(f' validator info: {validator}')
            total_balance += int(validator.balance)
            if validator.status in [ValidatorStatus.ACTIVE_ONGOING, ValidatorStatus.WITHDRAWAL_POSSIBLE,
                                    ValidatorStatus.ACTIVE_EXITING,
                                    ValidatorStatus.ACTIVE_SLASHED, ValidatorStatus.EXITED_SLASHED,
                                    ValidatorStatus.EXITED_UNSLASHED
                                    ]:
                # 需要上传激活的验证者数量
                validators_count += 1
                active_validators_balance += int(validator.balance)
            elif validator.status in ['withdrawal_done']:
                exited_validators_count += 1

        full_metrics.beaconBalance = total_balance
        full_metrics.activeValidatorBalance = active_validators_balance
        full_metrics.beaconValidators = validators_count
        full_metrics.exitedValidatorsCount = exited_validators_count

        logging.info(
            f'DawnPool validators\' beaconBalance: {full_metrics.beaconBalance},beaconValidators:{full_metrics.beaconValidators}'
            f',activeValidatorBalance:{full_metrics.activeValidatorBalance},exitedValidatorsCount:{full_metrics.exitedValidatorsCount}'
        )

        #  查询奖励库的地址当前时间对应账户在指定区块高度时的余额
        full_metrics.rewardsVaultBalance = w3.eth.get_balance(
            w3.to_checksum_address(rewards_vault_address.replace('0x010000000000000000000000', '0x')),
            block_identifier=block_number
        )
        logging.info(f'DawnPool the balance of the reward pool address : {full_metrics.rewardsVaultBalance}')

        # 获取lastRequestIdToBeFulfilled和ethAmountToLock
        buffered_ether = self.w3.lido_contracts.pool.functions.getBufferedEther().call()
        # 返回数组切片 returns (WithdrawRequest[] memory unfulfilledWithdrawRequestQueue)
        unfulfilled_withdraw_request_queue = self.w3.lido_contracts.withdraw.functions.getUnfulfilledWithdrawRequestQueue().call()
        logging.info(f'Dawn getUnfulfilledWithdrawRequestQueue : {unfulfilled_withdraw_request_queue}')

        request_sum = 0
        target_index = 0
        target_value = 0
        latest_index = 0

        logging.info(
            f'Dawn validators full_metrics: {full_metrics.beaconValidators}, {full_metrics.activeValidatorBalance},'
            f'{full_metrics.rewardsVaultBalance},{full_metrics.exitedValidatorsCount}')

        total_ether = 0
        total_peth = 0
        # 计算汇率：预估当前数据提交后，汇率是多少  没有奖励有异常进程停止，捕获异常
        # function preCalculateExchangeRate(uint256 beaconValidators, uint256 beaconBalance,uint256 availableRewards,
        # uint256 exitedValidators) external view returns (uint256 totalEther, uint256 totalPEth);
        try:
            total_ether, total_peth = self.w3.lido_contracts.pool.functions.preCalculateExchangeRate(
                full_metrics.beaconValidators,
                full_metrics.activeValidatorBalance,
                full_metrics.rewardsVaultBalance,
                full_metrics.exitedValidatorsCount).call()
        except ContractLogicError as e:
            logging.warning(f'Dawn get pre_calculate_exchange_rate total_ether,total_peth throw Exception: {str(e)} ')
            # 捕获异常,退出进程 ToDo
            # sys.exit()

        logging.info(f'Dawn pre_calculate_exchange_rate total_ether: {total_ether},total_peth: {total_peth}')
        # 遍历数组  从1开始遍历
        for i in range(1, len(unfulfilled_withdraw_request_queue)):
            if len(unfulfilled_withdraw_request_queue) < 2:
                break

            # struct WithdrawRequest {address owner;uint256 cumulativePEth; uint256 maxCumulativeClaimableEther;
            # uint256 createdTime; bool claimed; }
            # 赎回请求时汇率计算得到的ether
            eth_amount1 = unfulfilled_withdraw_request_queue[i][2] - unfulfilled_withdraw_request_queue[i - 1][2]
            logging.info(f'Dawn eth_amount1 : {eth_amount1}')
            # 赎回的peth量
            peth = unfulfilled_withdraw_request_queue[i][1] - unfulfilled_withdraw_request_queue[i - 1][1]
            logging.info(f'Dawn peth : {peth}, original eth_amount2: {peth * total_ether / total_peth}')
            # 按照当前汇率去计算 uint256 totalEther[0], uint256 totalPEth[1]
            # eth_amount2 = 0
            if total_peth == 0:
                eth_amount2 = 0
            else:
                eth_amount2 = decimal.Decimal(peth) * decimal.Decimal(total_ether) / decimal.Decimal(total_peth)

            if eth_amount2 == 0:
                eth_amount2_int = 0
            else:
                eth_amount2_int = int(eth_amount2.quantize(decimal.Decimal('1'), rounding=decimal.ROUND_DOWN))
            # eth_amount2 = peth * total_ether / total_peth
            logging.info(f'Dawn eth_amount2 : {eth_amount2}, eth_amount2_int: {eth_amount2_int}')
            actual_amount = min(eth_amount1, eth_amount2_int)
            logging.info(f'Dawn actual_amount : {actual_amount}')

            if request_sum + actual_amount > buffered_ether + full_metrics.rewardsVaultBalance:
                target_index = i - 1
                target_value = request_sum
                logging.info(
                    f'Dawn getUnfulfilledWithdrawRequestQueue  target_index: {i}, target_value: {target_value}')
                break
            request_sum += actual_amount
            target_index = len(unfulfilled_withdraw_request_queue) - 1
            target_value = request_sum

        logging.info(f'Dawn request_sum : {request_sum}')

        # returns (uint256 lastFulfillmentRequestId, uint256 lastRequestId, uint256 lastCheckpointIndex);
        withdraw_queue_stat = self.w3.lido_contracts.withdraw.functions.getWithdrawQueueStat().call()
        logging.info(
            f'Dawn withdraw_queue_stat : {withdraw_queue_stat[0]},{withdraw_queue_stat[1]},{withdraw_queue_stat[2]}')

        latest_index = withdraw_queue_stat[0] + target_index

        full_metrics.lastRequestIdToBeFulfilled = latest_index
        full_metrics.ethAmountToLock = target_value
        logging.info(f'Dawn latest_index : {latest_index},target_value: {target_value}, ')

        # 获取燃币金额
        burner_contract_to_burned = self.w3.lido_contracts.burner.functions.getPEthBurnRequest().call()
        withdraw_to_burned = unfulfilled_withdraw_request_queue[target_index][1] - unfulfilled_withdraw_request_queue[0][1]
        full_metrics.burnedPethAmount = burner_contract_to_burned + withdraw_to_burned
        logging.info(
            f'Dawn validators burnedPethAmount: {full_metrics.burnedPethAmount},burner_contract_to_burned:{burner_contract_to_burned},withdraw_to_burned:{withdraw_to_burned}')

        logging.info(f'DawnPool validators visible on Beacon: {full_metrics.beaconValidators}')
        return full_metrics

    @staticmethod
    def compare_pool_metrics(previous: PoolMetrics, current: PoolMetrics) -> bool:
        """Describes the economics of metrics change.
        Helps the Node operator to understand the effect of firing composed TX
        Returns true on suspicious metrics"""
        warnings = False
        assert previous.DEPOSIT_SIZE == current.DEPOSIT_SIZE
        DEPOSIT_SIZE = previous.DEPOSIT_SIZE
        # 间隔时间
        delta_seconds = current.timestamp - previous.timestamp
        # 新增验证者
        appeared_validators = current.beaconValidators - previous.beaconValidators
        logging.info(f'Time delta:  {delta_seconds} s')
        logging.info(
            f'depositedValidators before:{previous.depositedValidators} after:{current.depositedValidators} change:{current.depositedValidators - previous.depositedValidators}'
        )

        # 信标验证者数量意外减少
        if current.beaconValidators < previous.beaconValidators:
            # warnings = True
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logging.warning('The number of beacon validators unexpectedly decreased!')
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        logging.info(
            f'beaconValidators before:{previous.beaconValidators} after:{current.beaconValidators} change:{appeared_validators}'
        )
        logging.info(
            f'beaconBalance before:{previous.beaconBalance} after:{current.beaconBalance} change:{current.beaconBalance - previous.beaconBalance}'
        )
        logging.info(
            f'bufferedBalance before:{previous.bufferedBalance} after:{current.bufferedBalance} change:{current.bufferedBalance - previous.bufferedBalance}'
        )
        logging.info(f'activeValidatorBalance now:{current.activeValidatorBalance} ')

        # 验证者的币余额期望值(reward_base)  根据当前 epoch 中出现的有效验证者数量和每个验证者的抵押金额(DEPOSIT_SIZE) 计算出当前 epoch 的奖励基数 + 上一个epoch该验证者的余额
        reward_base = appeared_validators * DEPOSIT_SIZE + previous.activeValidatorBalance
        # 验证者当前epoch应该获得的奖励 = 当前epoch结束时验证者余额 - 验证者的币余额期望值
        reward = current.activeValidatorBalance - reward_base

        if not delta_seconds:
            logging.info('No time delta between current and previous epochs. Skip APR calculations.')
            assert reward == 0
            assert current.beaconValidators == previous.beaconValidators
            return

        # APR calculation
        if current.activeValidatorBalance == 0:
            daily_reward_rate = 0
        else:
            days = delta_seconds / 60 / 60 / 24
            daily_reward_rate = reward / current.activeValidatorBalance / days

        apr = daily_reward_rate * 365

        if reward >= 0:
            logging.info(f'Validators were rewarded {reward} wei or {reward / 1e18} ETH')
            logging.info(f'Daily staking reward rate for active validators: {daily_reward_rate * 100:.8f} %')
            logging.info(f'Staking APR for active validators: {apr * 100:.4f} %')
            if apr > current.MAX_APR:
                logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                logging.warning('Staking APR too high! Talk to your fellow oracles before submitting!')
                logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            if apr < current.MIN_APR:
                logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                logging.warning('Staking APR too low! Talk to your fellow oracles before submitting!')
                logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        else:
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logging.warning(f'Penalties will decrease totalPooledEther by {-reward} wei or {-reward / 1e18} ETH')
            logging.warning('Validators were either slashed or suffered penalties!')
            logging.warning('Talk to your fellow oracles before submitting!')
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if reward == 0:
            logging.info(
                'Beacon balances stay intact (neither slashed nor rewarded). So this report won\'t have any economical impact on the pool.'
            )

        return warnings

    def get_timestamp_by_epoch(self, beacon_spec, epoch_id):
        """Required to calculate time-bound values such as APR"""
        slots_per_epoch = beacon_spec[1]
        seconds_per_slot = beacon_spec[2]
        genesis_time = beacon_spec[3]
        return genesis_time + slots_per_epoch * seconds_per_slot * epoch_id

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
