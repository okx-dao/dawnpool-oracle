import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NewType, Tuple

from eth_typing import ChecksumAddress
from web3.module import Module

from src.providers.consensus.typings import Validator
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
class DawnPoolValidator(Validator):
    dawnpool_id: DawnPoolKey


class CountOfKeysDiffersException(Exception):
    pass


ValidatorsByNodeOperator = dict[NodeOperatorGlobalIndex, list[DawnPoolValidator]]








