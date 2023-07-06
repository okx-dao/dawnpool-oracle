import logging
import os
import json
from functools import reduce

from web3.contract import Contract
from web3_multi_provider import MultiProvider
from web3.types import Wei
from web3 import Web3

from src.constants import (
    CHURN_LIMIT_QUOTIENT,
    FAR_FUTURE_EPOCH,
    MAX_EFFECTIVE_BALANCE,
    MAX_SEED_LOOKAHEAD,
    MAX_WITHDRAWALS_PER_PAYLOAD,
    MIN_PER_EPOCH_CHURN_LIMIT,
    MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
)
from src.metrics.prometheus.business import CONTRACT_ON_PAUSE, FRAME_PREV_REPORT_REF_SLOT, FRAME_PREV_REPORT_REF_EPOCH
from src.metrics.prometheus.ejector import (
    EJECTOR_VALIDATORS_COUNT_TO_EJECT,
    EJECTOR_TO_WITHDRAW_WEI_AMOUNT,
    EJECTOR_MAX_EXIT_EPOCH
)
from src.metrics.prometheus.duration_meter import duration_meter
from src.modules.ejector.data_encode import encode_data
from src.modules.ejector.typings import EjectorProcessingState, ReportData, ReportEjectData
from src.modules.submodules.consensus import ConsensusModule
from src.modules.submodules.oracle_module import BaseModule, ModuleExecuteDelay
from src.providers.consensus.typings import Validator
from src.services.prediction import RewardsPredictionService
from src.services.validator_state import LidoValidatorStateService
from src.typings import BlockStamp, EpochNumber, ReferenceBlockStamp
from src.utils.abi import named_tuple_to_dataclass
from src.utils.cache import global_lru_cache as lru_cache
from src.utils.validator_state import (
    is_active_validator,
    is_fully_withdrawable_validator,
    is_partially_withdrawable_validator,
)
from src.web3py.extensions.lido_validators import LidoValidator, NodeOperatorGlobalIndex, DawnPoolValidator
from src.web3py.typings import Web3

logger = logging.getLogger(__name__)

# ARTIFACTS_DIR = './assets'
# ORACLE_ARTIFACT_FILE = 'DawnPoolOracle.json'
# POOL_ARTIFACT_FILE = 'DawnDeposit.json'
# REGISTRY_ARTIFACT_FILE = 'DepositNodeManager.json'
# WITHDRAW_ARTIFACT_FILE = 'DawnWithdraw.json'
#
# pool_address = os.environ['POOL_CONTRACT']
# if not Web3.is_checksum_address(pool_address):
#     pool_address = Web3.to_checksum_address(pool_address)
#
# oracle_address = os.environ['ORACLE_CONTRACT']
# if not Web3.is_checksum_address(oracle_address):
#     oracle_address = Web3.to_checksum_address(oracle_address)
#
# node_manager_address = os.environ['NODE_MANAGER_CONTRACT']
# if not Web3.is_checksum_address(node_manager_address):
#     node_manager_address = Web3.to_checksum_address(node_manager_address)
#
# withdraw_address = os.environ['WITHDRAW_CONTRACT']
# if not Web3.is_checksum_address(withdraw_address):
#     withdraw_address = Web3.to_checksum_address(withdraw_address)
#
# 奖励库的地址
rewards_vault_address = os.environ['REWARDS_VAULT_ADDRESS']
if not Web3.is_checksum_address(rewards_vault_address):
    rewards_vault_address = Web3.to_checksum_address(rewards_vault_address)


# 获取合约abi路径
# oracle_abi_path = os.path.join(ARTIFACTS_DIR, ORACLE_ARTIFACT_FILE)
# pool_abi_path = os.path.join(ARTIFACTS_DIR, POOL_ARTIFACT_FILE)
# registry_abi_path = os.path.join(ARTIFACTS_DIR, REGISTRY_ARTIFACT_FILE)
# withdraw_abi_path = os.path.join(ARTIFACTS_DIR, WITHDRAW_ARTIFACT_FILE)

eth1_provider = os.environ['EXECUTION_CLIENT_URI']

provider = MultiProvider(eth1_provider.split(','))
w3 = Web3(provider)

if not w3.is_connected():
    logging.error('ETH node connection error!')
    exit(1)

# Get Pool contract
# with open(pool_abi_path, 'r') as file:
#     a = file.read()
# abi = json.loads(a)
# pool = w3.eth.contract(abi=abi['abi'], address=pool_address)  # contract object
#
# with open(oracle_abi_path, 'r') as file:
#     a = file.read()
# abi = json.loads(a)
# oracle = w3.eth.contract(abi=abi['abi'], address=oracle_address)
#
# with open(registry_abi_path, 'r') as file:
#     a = file.read()
# abi = json.loads(a)
# registry = w3.eth.contract(abi=abi['abi'], address=node_manager_address)
#
# with open(withdraw_abi_path, 'r') as file:
#     a = file.read()
# abi = json.loads(a)
# withdraw = w3.eth.contract(abi=abi['abi'], address=withdraw_address)


class Ejector(BaseModule, ConsensusModule):
    """
    Module that ejects lido validators depends on total value of unfinalized withdrawal requests.

    Flow:
    1. Calculate withdrawals amount to cover with ETH.
    2. Calculate ETH rewards prediction per epoch.
    Loop:
        1. Calculate withdrawn epoch for last validator in "to eject" list.
        2. Calculate predicted rewards we get until last validator will be withdrawn.
        3. Check if validators to eject + predicted rewards and withdrawals + current balance is enough to finalize all withdrawal requests.
            - If True - eject all validators in list. End.
        4. Add new validator to "to eject" list.
        5. Recalculate withdrawn epoch.

    3. Decode lido validators into bytes and send report transaction
    """
    CONSENSUS_VERSION = 1
    CONTRACT_VERSION = 1

    AVG_EXPECTING_WITHDRAWALS_SWEEP_DURATION_MULTIPLIER = 0.5

    def __init__(self, w3: Web3):
        self.report_contract = w3.lido_contracts.validators_exit_bus_oracle
        super().__init__(w3)

        self.prediction_service = RewardsPredictionService(w3)
        self.validators_state_service = LidoValidatorStateService(w3)

    def refresh_contracts(self):
        self.report_contract = self.w3.lido_contracts.validators_exit_bus_oracle

    def execute_module(self, last_finalized_blockstamp: BlockStamp) -> ModuleExecuteDelay:
        # todo 怎么获取
        report_blockstamp = self.get_blockstamp_for_report(last_finalized_blockstamp)
        if not report_blockstamp:
            return ModuleExecuteDelay.NEXT_FINALIZED_EPOCH

        self.process_report(report_blockstamp)
        return ModuleExecuteDelay.NEXT_SLOT

    @lru_cache(maxsize=1)
    @duration_meter()
    def build_report(self, blockstamp: ReferenceBlockStamp) -> tuple:
        # last_report_ref_slot = self.w3.lido_contracts.get_ejector_last_processing_ref_slot(blockstamp)
        # FRAME_PREV_REPORT_REF_SLOT.set(last_report_ref_slot)
        last_report_ref_epoch = self.w3.lido_contracts.get_ejector_last_processing_ref_epoch(blockstamp)
        FRAME_PREV_REPORT_REF_EPOCH.set(last_report_ref_epoch)
        logger.info({'msg': 'build_report last_report_ref_epoch', 'value': last_report_ref_epoch})
        eject_count: int = self.get_validators_to_eject(blockstamp)
        logger.info({
            'msg': f'Calculate validators to eject. Count: {eject_count}'},
        )

        # data, data_format = encode_data(validators)

        report_data = ReportEjectData(
            blockstamp.ref_epoch,
            eject_count,
        )

        EJECTOR_VALIDATORS_COUNT_TO_EJECT.set(report_data.requests_count)

        return report_data.as_tuple()

    # todo 有些数据获取逻辑比较复杂 可以先从合约方法中获取
    # 计算需要弹出的验证器 接受一个区块时间戳对象 blockstamp 作为输入参数，并返回一个列表，包含需要弹出的验证器的全局索引和 Validator 类型的对象组成的元组。
    def get_validators_to_eject(self, blockstamp: ReferenceBlockStamp) -> int:
        # 所有未完成提现请求的总金额(需要退出的数量)，它将被用于计算可退出验证人的余额。 ToDo 调withdraw合约拿
        to_withdraw_amount = self.w3.lido_contracts.withdraw.functions.getUnfulfilledTotalEth().call(block_identifier=blockstamp.block_hash)
        logger.info({'msg': 'Calculate to withdraw amount.', 'value': to_withdraw_amount})

        # 可退出验证人的总余额（Wei）
        EJECTOR_TO_WITHDRAW_WEI_AMOUNT.set(to_withdraw_amount)
        # 如果 to_withdraw_amount 的值为 0，则表示没有未完成的提现请求需要处理，因此该函数直接返回0。
        if to_withdraw_amount == Wei(0):
            return 0
        # 获取给定块的链配置对象 slotsPerEpoch：每个纪元中的 slot 数量(32)； secondsPerSlot：每个 slot 的时长（12秒）；genesisTime：链的创世时间（Unix 时间戳）todo 通过HashConsensus合约获取()
        #  chain_config = self.get_chain_config(blockstamp)
        chain_config = self.w3.lido_contracts.validators_exit_bus_oracle.functions.getChainConfig().call()
        logger.info({'msg': 'Fetch chain config.', 'value': chain_config})
        # 每个 epoch 中，每个验证人可以获得多少奖励。 todo 通过上报事件平均值计算
        rewards_speed_per_epoch = self.prediction_service.get_rewards_per_epoch(blockstamp, chain_config)
        logger.info({'msg': 'Calculate average rewards speed per epoch.', 'value': rewards_speed_per_epoch})
        # 计算需要多少个 epoch 才能处理完链上所有需要进行提款的验证人 todo 查询链上数据获取 可以复用
        epochs_to_sweep = self._get_sweep_delay_in_epochs(blockstamp)
        # epochs_to_sweep = 1
        logger.info({'msg': 'Calculate epochs to sweep.', 'value': epochs_to_sweep})

        # 奖励库金额
        rewards_balance = w3.eth.get_balance(
            w3.to_checksum_address(rewards_vault_address.replace('0x010000000000000000000000', '0x')),
            block_identifier=blockstamp.block_number
        )
        # 合约中的 ETH 缓冲余额
        buffered_ether = self.w3.lido_contracts.pool.functions.getBufferedEther().call()
        # 获取指定块上当前可用的ETH总余额 = 奖励库金额 + 合约中的 ETH 缓冲余额
        total_available_balance = rewards_balance + buffered_ether
        logger.info({'msg': 'Calculate total_available_balance.', 'value': total_available_balance,
                     'rewards_balance': rewards_balance, 'buffered_ether': buffered_ether})

        # total_available_balance = self._get_total_el_balance(blockstamp)
        logger.info({'msg': 'Calculate el balance.', 'value': total_available_balance})
        # 获取最近提交了退出请求但尚未退出的验证人列表 todo
        validators_going_to_exit = self.validators_state_service.get_recently_requested_but_not_exited_validators(blockstamp, chain_config)
        # 计算正在退出验证人的待提款ETH余额总和
        going_to_withdraw_balance = sum(map(
            self._get_predicted_withdrawable_balance,
            validators_going_to_exit,
        ))
        logger.info({'msg': 'Calculate going to exit validators balance.', 'value': going_to_withdraw_balance})
        # NodeOperatorGlobalIndex 表示要退出的验证人的全局索引；第二个元素 LidoValidator，表示要退出的验证人的详细信息
        # validators_to_eject: list[tuple[NodeOperatorGlobalIndex, LidoValidator]] = []
        eject_count = 0
        # 即将退出的验证人的总待提款余额
        validator_to_eject_balance_sum = 0
        # 可以退出的验证人列表 调验证者合约拿到所有验证者列表 过滤状态为VALIDATING的列表 对应枚举索引值为2
        # function getNodeValidators(uint256 startIndex, uint256 amount) external view returns (address[] memory operators, bytes[] memory pubkeys, ValidatorStatus[] memory statuses);
        node_validators = self.w3.lido_contracts.registry.functions.getNodeValidators(0, 0).call()
        logger.info({'msg': 'node_validators.', 'value': node_validators})
        # 使用列表解析来查找验证者状态为VALIDATING并生成包含目标元素索引的新列表
        index_status_list = [index for index in range(len(node_validators[2])) if node_validators[2][index] == 2]
        validators_list = []
        # 可以退出的验证人列表(遍历状态为VALIDATING的验证者数组,得到状态为VALIDATING的验证者数组)
        for index in index_status_list:
            validators_list.append(node_validators[1][index])

        dawn_pool_validators_list = self.w3.lido_validators.get_dawn_pool_validators_by_keys(blockstamp, validators_list)

        for validator in dawn_pool_validators_list:
            # 获取待提款的的可提现时期，以便确定在何时可以提取这些代币  提款请求队列中的已退出验证人数量加上正在退出验证人数量再加上1 todo 可以复用 跑数据验证下
            withdrawal_epoch = self._get_predicted_withdrawable_epoch(blockstamp, eject_count + len(validators_going_to_exit) + 1)
            # 计算即将退出的验证人可以获得的未来奖励数量  从当前 Epoch 开始，到待提款的 EL 代币可以提现的 Epoch 结束，再加上扫描的 Epoch 数量，共有多少个 Epoch. blockstamp.ref_epoch 返回指定时间戳所在的引用时期编号
            # 然后将这个 Epoch 数量乘以每个 Epoch 的奖励速度，即可得到该验证人在未来可以获得的奖励数量
            future_rewards = (withdrawal_epoch + epochs_to_sweep - blockstamp.ref_epoch) * rewards_speed_per_epoch
            # 验证者退出之前所有验证者的"全部提款"的总额  （指定时间戳和提款时期的可提取 验证者余额） todo
            future_withdrawals = self._get_withdrawable_lido_validators_balance(blockstamp, withdrawal_epoch)

            # 计算当前的预期总余额
            expected_balance = (
                future_withdrawals +  # Validators that have withdrawal_epoch 指定时间戳和提款时期的可提取 验证者余额
                future_rewards +  # Rewards we get until last validator in validators_to_eject will be withdrawn todo 验证人在未来可以获得的奖励数量
                total_available_balance +  # Current EL balance (el vault, wc vault, buffered eth) 当前可用的ETH总余额
                validator_to_eject_balance_sum +  # Validators that we expected to be ejected (requested to exit, not delayed) 已请求退出但仍在延迟期内的验证人节点持有的ETH总余额之和
                going_to_withdraw_balance  # validators_to_eject balance 已被排队等待退出的验证人节点持有的 ETH
            )
            # 判断是否已经可以退出指定数量的验证人节点，并在可以退出的情况下返回要退出的验证人节点列表  预期总余额大于或等于需要退出的代币数量
            if expected_balance >= to_withdraw_amount:
                logger.info({
                    'msg': f'Expected withdrawal epoch: {withdrawal_epoch=}, '
                           f'will be reached in {withdrawal_epoch - blockstamp.ref_epoch} epochs. '
                           f'Validators with withdrawal_epoch before expected: {future_withdrawals=}. '
                           f'Future rewards from skimming and EL rewards: {future_rewards=}. '
                           f'Currently available balance: {total_available_balance=}. '
                           f'Validators expecting to start exit balance: {validator_to_eject_balance_sum=}. '
                           f'Validators going to eject balance: {going_to_withdraw_balance=}. ',
                    'withdrawal_epoch': withdrawal_epoch,
                    'ref_epoch': blockstamp.ref_epoch,
                    'future_withdrawals': future_withdrawals,
                    'future_rewards': future_rewards,
                    'total_available_balance': total_available_balance,
                    'validator_to_eject_balance_sum': validator_to_eject_balance_sum,
                    'going_to_withdraw_balance': going_to_withdraw_balance,
                })
                # 验证人列表：包含需要退出的验证人节点的地址和相应的代币数量。
                return eject_count
            # 将可以退出的验证人节点添加到validators_to_eject列表中  该元组包含了需要退出的验证人节点在验证人列表中的索引和验证人节点本身。
            # validators_to_eject.append(validator_container)
            # (_, validator) = validator_container
            # 更新已请求退出但仍在延迟期内的验证人节点持有的 EL 代币数量之和 todo 查询有效余额 取有效余额和32的最小值
            # validator_to_eject_balance_sum += MAX_EFFECTIVE_BALANCE
            validator_to_eject_balance_sum += self._get_predicted_withdrawable_balance(validator)
            eject_count += 1

        return eject_count

    def _is_paused(self, blockstamp: ReferenceBlockStamp) -> bool:
        return self.report_contract.functions.isPaused().call(block_identifier=blockstamp.block_hash)

    def is_reporting_allowed(self, blockstamp: ReferenceBlockStamp) -> bool:
        on_pause = self._is_paused(blockstamp)
        CONTRACT_ON_PAUSE.set(on_pause)
        logger.info({'msg': 'Fetch isPaused from ejector bus contract.', 'value': on_pause})
        return not on_pause

    @lru_cache(maxsize=1)
    def _get_withdrawable_lido_validators_balance(self, blockstamp: BlockStamp, on_epoch: EpochNumber) -> Wei:
        # 获取lido系统中所有的验证人列表
        #   lido_validators = self.w3.lido_validators.get_lido_validators(blockstamp=blockstamp)

        dawn_pool_validators = []
        # node_validators = registry.functions.getNodeValidators(0, 0).call()[1]

        # event SigningKeyExiting(uint256 indexed validatorId, address indexed operator, bytes pubkey); 最近请求退出验证者的事件
        exiting_events = self.w3.lido_contracts.registry.events.SigningKeyExiting.getLogs()
        dawn_pool_exiting_list = []
        for event in exiting_events:
            dawn_pool_exiting_list.append(DawnPoolValidator(event.pubkey, event.validatorId, event.operator))
            #     ToDo 根据pubkey去链上查询状态 复用lido共识层拿数据
        dawn_pool_validators = self.w3.lido_validators.get_dawn_pool_validators(blockstamp)

        for exiting in exiting_events:
            for validator in dawn_pool_validators:
                # pubkey相等
                if exiting['pubkey'] == validator['pubkey']:
                    dawn_pool_validators.append({
                        **exiting['args'],
                        **validator['args'],
                    })
                    break

        def get_total_withdrawable_balance(balance: Wei, validator: Validator) -> Wei:
            # 在on_epoch 是否可以提款
            if is_fully_withdrawable_validator(validator, on_epoch):
                # 预测可提现余额
                balance = Wei(
                    balance + self._get_predicted_withdrawable_balance(validator)
                )

            return balance

        result = reduce(
            get_total_withdrawable_balance,
            dawn_pool_validators,
            Wei(0),
        )

        return result

    # 计算验证人的预测可提取余额  一个验证人的预测可提取余额是指，如果该验证人现在退出Lido验证人池，那么他可以从Lido合约中提取出的有效质押金额。
    def _get_predicted_withdrawable_balance(self, validator: Validator) -> Wei:
        return self.w3.to_wei(min(int(validator.balance), MAX_EFFECTIVE_BALANCE), 'gwei')

    # todo 合约中获取 查询合约余额
    def _get_total_el_balance(self, blockstamp: BlockStamp) -> Wei:
        total_el_balance = Wei(
            #  合约中的余额
            self.w3.lido_contracts.get_el_vault_balance(blockstamp) +
            # 已提交但尚未处理的提款请求的代币金额
            self.w3.lido_contracts.get_withdrawal_balance(blockstamp) +
            #  合约中的 ETH 缓冲余额 ToDo 查询合约余额 getBufferedEther
            self._get_buffer_ether(blockstamp)
        )
        return total_el_balance

    def _get_buffer_ether(self, blockstamp: BlockStamp) -> Wei:
        """
        The reserved buffered ether is min(current_buffered_ether, unfinalized_withdrawal_requests_amount)
        We can skip calculating reserved buffer for ejector, because in case if
        (unfinalized_withdrawal_requests_amount <= current_buffered_ether)
        We won't eject validators at all, because we have enough eth to fulfill all requests.
        """
        return Wei(
            self.w3.lido_contracts.lido.functions.getBufferedEther().call(
                block_identifier=blockstamp.block_hash
            )
        )

    def get_total_unfinalized_withdrawal_requests_amount(self, blockstamp: BlockStamp) -> Wei:
        # 它使用了 self.w3.lido_contracts.withdrawal_queue_nft 获取一个与 WithdrawalsStoreNFT 合约交互的合约对象，并调用了该合约对象的 unfinalizedStETH 方法获取未完成提现请求的总金额。
        unfinalized_steth = self.w3.lido_contracts.withdrawal_queue_nft.functions.unfinalizedStETH().call(
            block_identifier=blockstamp.block_hash,
        )
        return unfinalized_steth

    # 计算出所有在队列中以及即将退出的验证者可以撤回奖励的 Epoch
    def _get_predicted_withdrawable_epoch(
        self,
        blockstamp: ReferenceBlockStamp,
        validators_to_eject_count: int,
    ) -> EpochNumber:
        """
        Returns epoch when all validators in queue and validators_to_eject will be withdrawn.
        """
        max_exit_epoch_number, latest_to_exit_validators_count = self._get_latest_exit_epoch(blockstamp)

        max_exit_epoch_number = max(
            max_exit_epoch_number,
            self.compute_activation_exit_epoch(blockstamp),
        )

        EJECTOR_MAX_EXIT_EPOCH.set(max_exit_epoch_number)

        churn_limit = self._get_churn_limit(blockstamp)

        remain_exits_capacity_for_epoch = churn_limit - latest_to_exit_validators_count
        epochs_required_to_exit_validators = (validators_to_eject_count - remain_exits_capacity_for_epoch) // churn_limit + 1

        return EpochNumber(max_exit_epoch_number + epochs_required_to_exit_validators + MIN_VALIDATOR_WITHDRAWABILITY_DELAY)

    @staticmethod
    def compute_activation_exit_epoch(blockstamp: ReferenceBlockStamp):
        """
        Return the epoch during which validator activations and exits initiated in ``epoch`` take effect.

        Spec: https://github.com/LeastAuthority/eth2.0-specs/blob/dev/specs/phase0/beacon-chain.md#compute_activation_exit_epoch
        """
        return blockstamp.ref_epoch + 1 + MAX_SEED_LOOKAHEAD

    @lru_cache(maxsize=1)
    def _get_latest_exit_epoch(self, blockstamp: BlockStamp) -> tuple[EpochNumber, int]:
        """
        Returns the latest exit epoch and amount of validators that are exiting in this epoch
        """
        max_exit_epoch_number = EpochNumber(0)
        latest_to_exit_validators_count = 0

        for validator in self.w3.cc.get_validators(blockstamp):
            val_exit_epoch = EpochNumber(int(validator.validator.exit_epoch))

            if val_exit_epoch == FAR_FUTURE_EPOCH:
                continue

            if val_exit_epoch == max_exit_epoch_number:
                latest_to_exit_validators_count += 1

            elif val_exit_epoch > max_exit_epoch_number:
                max_exit_epoch_number = val_exit_epoch
                latest_to_exit_validators_count = 1

        return max_exit_epoch_number, latest_to_exit_validators_count

    # 计算需要多少个 epoch 才能处理完链上所有需要进行提款的验证人
    def _get_sweep_delay_in_epochs(self, blockstamp: ReferenceBlockStamp) -> int:
        """Returns amount of epochs that will take to sweep all validators in chain."""
        # 获取当前所有的验证人
        validators = self.w3.cc.get_validators(blockstamp)

        # filter 函数筛选出所有可以进行提款的验证人，即部分提款或者全部提款都可以进行的验证人
        total_withdrawable_validators = len(list(filter(lambda validator: (
            # 判断验证人是否可以进行部分提款或者全部提款
            is_partially_withdrawable_validator(validator) or
            is_fully_withdrawable_validator(validator, blockstamp.ref_epoch)
        ), validators)))

        chain_config = self.get_chain_config(blockstamp)
        # MAX_WITHDRAWALS_PER_PAYLOAD 表示每次向链上提交的提款交易数量的上限
        full_sweep_in_epochs = total_withdrawable_validators / MAX_WITHDRAWALS_PER_PAYLOAD / chain_config.slotsPerEpoch
        # 计算出的需要清理的 epoch 数量
        return int(full_sweep_in_epochs * self.AVG_EXPECTING_WITHDRAWALS_SWEEP_DURATION_MULTIPLIER)

    @lru_cache(maxsize=1)
    def _get_churn_limit(self, blockstamp: ReferenceBlockStamp) -> int:
        total_active_validators = reduce(
            lambda total, validator: total + int(is_active_validator(validator, blockstamp.ref_epoch)),
            self.w3.cc.get_validators(blockstamp),
            0,
        )
        return max(MIN_PER_EPOCH_CHURN_LIMIT, total_active_validators // CHURN_LIMIT_QUOTIENT)

    def _get_processing_state(self, blockstamp: BlockStamp) -> EjectorProcessingState:
        ps = named_tuple_to_dataclass(
            self.report_contract.functions.getProcessingState().call(block_identifier=blockstamp.block_hash),
            EjectorProcessingState,
        )
        logger.info({'msg': 'Fetch processing state.', 'value': ps})
        return ps

    def is_main_data_submitted(self, blockstamp: BlockStamp) -> bool:
        processing_state = self._get_processing_state(blockstamp)
        return processing_state.data_submitted

    def is_contract_reportable(self, blockstamp: BlockStamp) -> bool:
        return not self.is_main_data_submitted(blockstamp)
