import logging
import binascii

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NewType, Tuple, Optional

from eth_typing import ChecksumAddress
from web3.contract import Contract
from web3.module import Module

from src.providers.consensus.typings import Validator
from src.providers.keys.typings import LidoKey
from src.providers.keys.typings import DawnPoolKey
from src.typings import BlockStamp
from src.utils.dataclass import Nested, list_of_dataclasses
from src.utils.cache import global_lru_cache as lru_cache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.web3py.typings import Web3  # pragma: no cover

StakingModuleId = NewType('StakingModuleId', int)
NodeOperatorId = NewType('NodeOperatorId', int)
NodeOperatorGlobalIndex = Tuple[StakingModuleId, NodeOperatorId]


@dataclass
class StakingModule:
    # unique id of the staking module
    id: StakingModuleId
    # address of staking module
    staking_module_address: ChecksumAddress
    # part of the fee taken from staking rewards that goes to the staking module
    staking_module_fee: int
    # part of the fee taken from staking rewards that goes to the treasury
    treasury_fee: int
    # target percent of total validators in protocol, in BP
    target_share: int
    # staking module status if staking module can not accept
    # the deposits or can participate in further reward distribution
    status: int
    # name of staking module
    name: str
    # block.timestamp of the last deposit of the staking module
    last_deposit_at: int
    # block.number of the last deposit of the staking module
    last_deposit_block: int
    # number of exited validators
    exited_validators_count: int


@dataclass
class NodeOperator(Nested):
    id: NodeOperatorId
    is_active: bool
    is_target_limit_active: bool
    target_validators_count: int
    stuck_validators_count: int
    refunded_validators_count: int
    stuck_penalty_end_timestamp: int
    total_exited_validators: int
    total_deposited_validators: int
    depositable_validators_count: int
    staking_module: StakingModule

    @classmethod
    def from_response(cls, data, staking_module):
        _id, is_active, (
            is_target_limit_active,
            target_validators_count,
            stuck_validators_count,
            refunded_validators_count,
            stuck_penalty_end_timestamp,
            total_exited_validators,
            total_deposited_validators,
            depositable_validators_count,
        ) = data

        return cls(
            _id,
            is_active,
            is_target_limit_active,
            target_validators_count,
            stuck_validators_count,
            refunded_validators_count,
            stuck_penalty_end_timestamp,
            total_exited_validators,
            total_deposited_validators,
            depositable_validators_count,
            staking_module,
        )


@dataclass
class LidoValidator(Validator):
    lido_id: LidoKey


@dataclass
class DawnPoolValidator(Validator):
    dawnpool_id: DawnPoolKey


class CountOfKeysDiffersException(Exception):
    pass


ValidatorsByNodeOperator = dict[NodeOperatorGlobalIndex, list[LidoValidator]]


class LidoValidatorsProvider(Module):
    w3: 'Web3'

    @lru_cache(maxsize=1)
    def get_lido_validators(self, blockstamp: BlockStamp) -> list[LidoValidator]:
        lido_keys = self.w3.kac.get_used_lido_keys(blockstamp)
        validators = self.w3.cc.get_validators(blockstamp)

        no_operators = self.get_lido_node_operators(blockstamp)

        # Make sure that used keys fetched from Keys API >= total amount of total deposited validators from Staking Router
        total_deposited_validators = 0
        for deposited_validators in no_operators:
            total_deposited_validators += deposited_validators.total_deposited_validators

        if len(lido_keys) < total_deposited_validators:
            raise CountOfKeysDiffersException(f'Keys API Service returned lesser keys ({len(lido_keys)}) '
                                              f'than amount of deposited validators ({total_deposited_validators}) returned from Staking Router')

        return self.merge_validators_with_keys(lido_keys, validators)

    @lru_cache(maxsize=1)
    def get_dawn_pool_validators(self, blockstamp: BlockStamp) -> list[DawnPoolValidator]:
        # 通过我们合约获取所有pub_keys todo
        dawn_pool_keys = self.w3.lido_contracts.registry.functions.getNodeValidators(0, 0).call()[1]

        hex_keys = tuple('0x' + binascii.hexlify(pk).decode('ascii') for pk in dawn_pool_keys)
        logger.info({'msg': 'Fetch dawn pool keys.', 'value': hex_keys})

        validators = self.w3.cc.get_pub_key_validators(blockstamp, hex_keys)

        logger.info({'msg': 'Fetch dawn pool validators.', 'value': validators})
        return self.merge_dawn_pool_validators_with_keys(validators, hex_keys)

    # @lru_cache(maxsize=1)
    # def get_dawn_pool_validators_by_keys(self, blockstamp: BlockStamp, pub_keys: Optional[str | tuple] = None) -> list[DawnPoolValidator]:
    #     logger.info({'msg': 'Fetch dawn pool validators pub_keys.', 'value': pub_keys})
    #     validators = self.w3.cc.get_pub_key_validators(blockstamp, pub_keys)
    #     logger.info({'msg': 'Fetch dawn_pool_validators_by_keys.', 'value': validators})
    #     return self.merge_dawn_pool_validators_with_keys(pub_keys, validators)

    @lru_cache(maxsize=1)
    def get_dawn_pool_validators_by_keys(self, blockstamp: BlockStamp) -> list[
        DawnPoolValidator]:

        # function getNodeValidators(uint256 startIndex, uint256 amount) external view returns (address[] memory operators, bytes[] memory pubkeys, ValidatorStatus[] memory statuses);
        node_validators = self.w3.lido_contracts.registry.functions.getNodeValidators(0, 0).call()

        logger.info({'msg': 'node_validators.', 'value': node_validators})
        # 使用列表解析来查找验证者状态为VALIDATING并生成包含目标元素索引的新列表
        index_status_list = [index for index in range(len(node_validators[2])) if node_validators[2][index] == 2]
        validators_list = []
        # 可以退出的验证人列表(遍历状态为VALIDATING的验证者数组,得到状态为VALIDATING的验证者数组)
        for index in index_status_list:
            validators_list.append(node_validators[1][index])

        logger.info(
            {'msg': 'Fetch dawn pool validators pub_keys.', 'len': len(index_status_list), 'value': index_status_list})

        hex_status_keys = tuple('0x' + binascii.hexlify(pk).decode('ascii') for pk in validators_list)
        logger.info({'msg': 'Fetch dawn pool status keys.', 'value': hex_status_keys})

        validators = self.w3.cc.get_pub_key_validators(blockstamp, hex_status_keys)

        logger.info({'msg': 'Fetch dawn_pool_validators_by_keys.', 'value': validators})
        return self.merge_dawn_pool_validators_with_keys(validators, hex_status_keys)

    @staticmethod
    def merge_validators_with_keys(keys: list[LidoKey], validators: list[Validator]) -> list[LidoValidator]:
        """Merging and filter non-lido validators."""
        validators_keys_dict = {validator.validator.pubkey: validator for validator in validators}

        lido_validators = []

        for key in keys:
            if key.key in validators_keys_dict:
                lido_validators.append(LidoValidator(
                    lido_id=key,
                    **asdict(validators_keys_dict[key.key]),
                ))

        return lido_validators

    @staticmethod
    def merge_dawn_pool_validators_with_keys(validators: list[Validator], keys: Optional[str | tuple] = None) -> list[
        DawnPoolValidator]:
        """Merging and filter non-lido validators."""

        logger.info({'msg': 'Fetch merge_dawn_pool_validators_with_keys.', 'value': keys})
        validators_keys_dict = {validator.validator.pubkey: validator for validator in validators}
        logger.info({'msg': 'Fetch validators_keys_dict.', 'value': validators_keys_dict})
        dawn_pool_validators = []

        for key in keys:
            logger.info({'msg': 'Fetch merge_dawn_pool_validators_with_keys key.', 'value': key})
            if key in validators_keys_dict:
                dawn_pool_validators.append(DawnPoolValidator(
                    dawnpool_id=DawnPoolKey(key, 0, '0x00'),
                    **asdict(validators_keys_dict[key]),
                ))

        logger.info({'msg': 'Fetch dawn_pool_validators.', 'value': dawn_pool_validators})
        return dawn_pool_validators

    @lru_cache(maxsize=1)
    def get_lido_validators_by_node_operators(self, blockstamp: BlockStamp) -> ValidatorsByNodeOperator:
        merged_validators = self.get_lido_validators(blockstamp)
        no_operators = self.get_lido_node_operators(blockstamp)

        # Make sure even empty NO will be presented in dict
        no_validators: ValidatorsByNodeOperator = {
            (operator.staking_module.id, operator.id): [] for operator in no_operators
        }

        staking_module_address = {
            operator.staking_module.staking_module_address: operator.staking_module.id
            for operator in no_operators
        }

        for validator in merged_validators:
            global_no_id = (
                staking_module_address[validator.lido_id.moduleAddress],
                NodeOperatorId(validator.lido_id.operatorIndex),
            )

            if global_no_id in no_validators:
                no_validators[global_no_id].append(validator)
            else:
                logger.warning({
                    'msg': f'Got global node operator id: {global_no_id}, '
                           f'but it`s not exist in staking router on block number: {blockstamp.block_number}',
                })

        return no_validators

    @lru_cache(maxsize=1)
    def get_lido_node_operators(self, blockstamp: BlockStamp) -> list[NodeOperator]:
        result = []

        for module in self.get_staking_modules(blockstamp):
            operators = self.w3.lido_contracts.staking_router.functions.getAllNodeOperatorDigests(
                module.id
            ).call(block_identifier=blockstamp.block_hash)

            for operator in operators:
                result.append(NodeOperator.from_response(operator, module))

        return result

    @lru_cache(maxsize=1)
    @list_of_dataclasses(StakingModule)
    def get_staking_modules(self, blockstamp: BlockStamp) -> list[StakingModule]:
        modules = self.w3.lido_contracts.staking_router.functions.getStakingModules().call(
            block_identifier=blockstamp.block_hash,
        )

        logger.info({'msg': 'Fetch staking modules.', 'value': modules})

        return modules
