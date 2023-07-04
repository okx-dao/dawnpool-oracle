import logging

from web3.contract import Contract
from web3.types import Wei, EventData

from src.modules.submodules.typings import ChainConfig
from src.typings import ReferenceBlockStamp
from src.utils.events import get_events_in_past
from src.web3py.typings import Web3


logger = logging.getLogger(__name__)


class RewardsPredictionService:
    """
    Based on events predicts amount of eth that protocol will earn per epoch.

    **Note** Withdraw amount in Oracle report is limited, so prediction shows not actual Lido rewards, but medium.
    amount of ETH that were withdrawn in Oracle reports.
    """
    def __init__(self, w3: Web3):
        self.w3 = w3

    # 根据当前 epoch 的链配置信息和计算出来的总奖励金额，计算出每个验证人可以获得的奖励速度。
    def get_rewards_per_epoch(
        self,
        blockstamp: ReferenceBlockStamp,
        chain_configs: ChainConfig,
    ) -> Wei:

        # 获取预测周期的时间戳数量
        # prediction_duration_in_slots = self._get_prediction_duration_in_slots(blockstamp)
        # logger.info({'msg': 'Fetch prediction frame in slots.', 'value': prediction_duration_in_slots})
        #
        # # 获取历史事件列表  仁贵提供
        # token_rebase_events = get_events_in_past(
        #     self.w3.lido_contracts.lido.events.TokenRebased,  # type: ignore[arg-type]
        #     blockstamp,
        #     prediction_duration_in_slots,
        #     chain_configs.seconds_per_slot,
        #     'reportTimestamp',
        # )
        #
        # #  仁贵提供
        # eth_distributed_events = get_events_in_past(
        #     self.w3.lido_contracts.lido.events.ETHDistributed,  # type: ignore[arg-type]
        #     blockstamp,
        #     prediction_duration_in_slots,
        #     chain_configs.seconds_per_slot,
        #     'reportTimestamp',
        # )
        #
        # events = self._group_events_by_transaction_hash(token_rebase_events, eth_distributed_events)

        # if not events:
        #     return Wei(0)

        # total_rewards = 0
        # time_spent = 0

        # for event in events:
        #     #  total_rewards += event['postCLBalance'] + event['rewardVault'] - event['preCLBalance']
        #     total_rewards += event['postCLBalance'] + event['withdrawalsWithdrawn'] - event['preCLBalance'] + event['executionLayerRewardsWithdrawn']

        #     time_spent += event['timeElapsed']


        total_rewards = 0
        # event LogETHRewards(uint256 epochId, uint256 preCLBalance, uint256 postCLBalance, uint256 rewardsVaultBalance);
        eth_rewards_events = self.w3.lido_contracts.pool.events.LogETHRewards.get_logs()
        logger.info({'msg': 'Fetch eth rewards events.', 'value': eth_rewards_events})

        if not eth_rewards_events:
            return Wei(0)

        if len(eth_rewards_events) > 7:
            # 使用 sorted() 函数对数组 eth_rewards_events 进行排序，通过 key 参数指定按照 epochId 字段进行排序,设置 reverse=True 参数来实现降序排序
            sorted_array = sorted(eth_rewards_events, key=lambda x: x.epochId, reverse=True)
            # 使用切片操作 [:7] 取出排序后的前 7 个元素
            eth_rewards_events = sorted_array[:7]

        #  拿到最大和最小的epochId 相减得到 time_spent
        max_epoch_id = max(eth_rewards_events, key=lambda x: x['epochId'])['epochId']
        min_epoch_id = min(eth_rewards_events, key=lambda x: x['epochId'])['epochId']

        time_spent = max_epoch_id - min_epoch_id

        for event in eth_rewards_events:
            total_rewards += event['postCLBalance'] + event['rewardVault'] - event['preCLBalance']

        return max(
            Wei(total_rewards * chain_configs.seconds_per_slot * chain_configs.slots_per_epoch // time_spent),
            Wei(0),
        )

    @staticmethod
    def _group_events_by_transaction_hash(event_type_1: list[EventData], event_type_2: list[EventData]):
        result_event_data = []

        for event_1 in event_type_1:
            for event_2 in event_type_2:
                if event_2['transactionHash'] == event_1['transactionHash']:
                    result_event_data.append({
                        **event_1['args'],
                        **event_2['args'],
                    })
                    break

        if len(event_type_1) == len(event_type_2) == len(result_event_data):
            return result_event_data

        raise ValueError(
            f"Events are inconsistent: {len(event_type_1)=}, {len(event_type_2)=}, {len(result_event_data)=}"
        )

    def _get_prediction_duration_in_slots(self, blockstamp: ReferenceBlockStamp) -> int:
        return Web3.to_int(
            self.w3.lido_contracts.oracle_daemon_config.functions.get('PREDICTION_DURATION_IN_SLOTS').call(
                block_identifier=blockstamp.block_hash,
            )
        )
