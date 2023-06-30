# SPDX-FileCopyrightText: 2020 Lido <info@lido.fi>

# SPDX-License-Identifier: GPL-3.0

import logging
import datetime

from web3 import Web3

from contracts import get_validators_keys
from pool_metrics import PoolMetrics
from prometheus_metrics import metrics_exporter_state


def get_previous_metrics(w3, pool, oracle, beacon_spec, rewards_vault_address, from_block=0) -> PoolMetrics:
    """Since the contract lacks a method that returns the time of last report and the reported numbers
    we are using web3.py filtering to fetch it from the contract events."""
    logging.info('Getting previously reported numbers (will be fetched from events)...')
    genesis_time = beacon_spec[3]
    result = PoolMetrics()
    # 通过abi调用pool合约 todo preDepositValidators 参数验证
    result.preDepositValidators, result.depositedValidators, result.beaconValidators, result.beaconBalance = pool.functions.getBeaconStat().call()
    # 缓冲余额(将缓冲的 eth 存入质押合约并将分块存款分配给节点运营商)
    result.bufferedBalance = pool.functions.getBufferedEther().call()
    # Calculate the earliest block to limit scanning depth 计算最早的块以限制扫描深度
    # 每个 ETH1 块的秒数 正常12s 出一个块 设置14s为了遍历
    SECONDS_PER_ETH1_BLOCK = 14
    latest_block = w3.eth.getBlock('latest')
    from_block = max(from_block, int((latest_block['timestamp'] - genesis_time) / SECONDS_PER_ETH1_BLOCK))

    latest_num = latest_block['number']
    logging.info(f'DawnPool from_block : {from_block}, latest_num : {latest_num}')
    # Try to fetch and parse last 'Completed' event from the contract. 遍历从合约中获取并解析最后一个“已完成”事件。
    step = 1000
    for end in range(latest_block['number'], from_block, -step):
        start = max(end - step + 1, from_block)
        # 调用了 getLogs 方法来读取区块链上合约中 Completed 事件在指定区块高度范围内的日志,日志会被存储在 events 变量中
        events = oracle.events.Completed.getLogs(fromBlock=start, toBlock=end)
        # 判断 events 是否为空 如果存在符合条件的事件日志，获取最后一个事件，即 events[-1]，并从中提取出相应的信息
        if events:
            logging.info(f'DawnPool events : {events}')
            event = events[-1]
            result.epoch = event['args']['epochId']
            result.blockNumber = event.blockNumber
            break

    #  查询奖励库的地址(配置在环境变量里)对应账户在指定区块高度时的余额
    result.rewardsVaultBalance = w3.eth.get_balance(
        w3.toChecksumAddress(rewards_vault_address.replace('0x010000000000000000000000', '0x')),
        block_identifier=result.blockNumber
    )
    logging.info(f'DawnPool result : {result}')
    # If the epoch has been assigned from the last event (not the first run) 如果纪元是从最后一个事件（不是第一次运行）分配的
    if result.epoch:
        result.timestamp = get_timestamp_by_epoch(beacon_spec, result.epoch)
    else:
        # If it's the first run, we set timestamp to genesis time 如果是第一次运行，我们将时间戳设置为创世时间
        result.timestamp = genesis_time
    return result


def get_light_current_metrics(w3, beacon, pool, oracle, beacon_spec):
    """Fetch current frame, buffered balance and epoch"""
    # 每帧的epoch数
    epochs_per_frame = beacon_spec[0]
    partial_metrics = PoolMetrics()
    partial_metrics.blockNumber = w3.eth.getBlock('latest')['number']  # Get the epoch that is finalized and reportable
    # 当前帧信息的数组  通过abi合约调用查询
    current_frame = oracle.functions.getCurrentFrame().call()
    # 当前帧所在的epoch，作为潜在的报告epoch
    potentially_reportable_epoch = current_frame[0]
    logging.info(f'Potentially reportable epoch: {potentially_reportable_epoch} (from ETH1 contract)')
    # 获得最终确定的纪元
    finalized_epoch_beacon = beacon.get_finalized_epoch()
    # For Web3 client
    # finalized_epoch_beacon = int(beacon.get_finality_checkpoint()['data']['finalized']['epoch'])
    logging.info(f'Last finalized epoch: {finalized_epoch_beacon} (from Beacon)')
    # //是向下取整  第二个通过计算得出的实际报告时代 计算信标链中已经最终化的时代数 finalized_epoch_beacon 所在的当前帧
    partial_metrics.epoch = min(
        potentially_reportable_epoch, (finalized_epoch_beacon // epochs_per_frame) * epochs_per_frame
    )
    partial_metrics.timestamp = get_timestamp_by_epoch(beacon_spec, partial_metrics.epoch)
    partial_metrics.depositedValidators = pool.functions.getBeaconStat().call()[0]
    partial_metrics.bufferedBalance = pool.functions.getBufferedEther().call()
    logging.info(f'Last partial_metrics: {partial_metrics} ')
    return partial_metrics


def get_full_current_metrics(
        w3: Web3, pool, registry, burner, withdraw, beacon, beacon_spec, partial_metrics, rewards_vault_address
) -> PoolMetrics:
    """The oracle fetches all the required states from ETH1 and ETH2 (validator balances)"""
    slots_per_epoch = beacon_spec[1]
    logging.info(f'Reportable slots_per_epoch: {slots_per_epoch} ,partial_metrics.epoch: {partial_metrics.epoch}')
    slot = partial_metrics.epoch * slots_per_epoch
    logging.info(f'Reportable state, epoch:{partial_metrics.epoch} slot:{slot}')
    #  获取注册的验证者的key 通过abi获取
    validators_keys = registry.functions.getNodeValidators(0, 0).call()[1]
    logging.info(f'Total validator keys detail: {validators_keys}')
    logging.info(f'Total validator keys in registry: {len(validators_keys)}')
    full_metrics = partial_metrics
    # 根据验证者的key在信标链中计算信标余额，信标验证器，活跃的验证者余额
    full_metrics.validatorsKeysNumber = len(validators_keys)
    (
        full_metrics.beaconBalance,
        full_metrics.beaconValidators,
        full_metrics.activeValidatorBalance,
        full_metrics.exitedValidatorsCount,
    ) = beacon.get_balances(slot, validators_keys)

    logging.info(
        f'DawnPool validators\' sum. balance on Beacon: '
        f'{full_metrics.beaconBalance} wei or {full_metrics.beaconBalance / 1e18} ETH'
    )

    block_number = beacon.get_block_by_beacon_slot(slot)

    logging.info(f'Validator block_number: {block_number}')
    #  查询奖励库的地址当前时间对应账户在指定区块高度时的余额
    full_metrics.rewardsVaultBalance = w3.eth.get_balance(
        w3.toChecksumAddress(rewards_vault_address.replace('0x010000000000000000000000', '0x')),
        block_identifier=block_number
    )
    logging.info(f'DawnPool the balance of the reward pool address : {full_metrics.rewardsVaultBalance}')

    # 新增获取燃币金额 todo
    full_metrics.burnedPethAmount = burner.functions.getPEthBurnRequest().call()
    logging.info(f'Dawn validators burnedPethAmount: {full_metrics.burnedPethAmount}')

    # 获取lastRequestIdToBeFulfilled和ethAmountToLock todo
    buffered_ether = pool.functions.getBufferedEther().call()
    # 返回数组切片 returns (WithdrawRequest[] memory unfulfilledWithdrawRequestQueue)
    unfulfilled_withdraw_request_queue = withdraw.functions.getUnfulfilledWithdrawRequestQueue().call()
    logging.info(f'Dawn getUnfulfilledWithdrawRequestQueue : {unfulfilled_withdraw_request_queue}')

    request_sum = 0
    target_index = 0
    target_value = 0
    latest_index = 0

    logging.info(f'Dawn validators full_metrics: {full_metrics.beaconValidators}, {full_metrics.activeValidatorBalance},'
                 f'{full_metrics.withdrawalVaultBalance},{full_metrics.exitedValidatorsCount}')

    # 计算汇率：预估当前数据提交后，汇率是多少
    # function preCalculateExchangeRate(uint256 beaconValidators, uint256 beaconBalance,uint256 availableRewards,
    # uint256 exitedValidators) external view returns (uint256 totalEther, uint256 totalPEth);
    total_ether, total_peth = pool.functions.preCalculateExchangeRate(full_metrics.beaconValidators,
                                                                      full_metrics.activeValidatorBalance,
                                                                      full_metrics.withdrawalVaultBalance,
                                                                      full_metrics.exitedValidatorsCount).call()
    logging.info(f'Dawn pre_calculate_exchange_rate : {total_ether},{total_peth}')
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
        logging.info(f'Dawn peth : {peth}')
        # 按照当前汇率去计算 uint256 totalEther[0], uint256 totalPEth[1]
        eth_amount2 = peth * total_ether / total_peth
        logging.info(f'Dawn eth_amount2 : {eth_amount2}')
        actual_amount = min(eth_amount1, eth_amount2)
        logging.info(f'Dawn actual_amount : {actual_amount}')

        request_sum += actual_amount

        if request_sum > buffered_ether + full_metrics.withdrawalVaultBalance:
            target_index = i - 1
            target_value = request_sum
            logging.info(f'Dawn getUnfulfilledWithdrawRequestQueue  target_index: {i}, target_value: {target_value}')
            break
        target_index = len(unfulfilled_withdraw_request_queue) - 1

    logging.info(f'Dawn request_sum : {request_sum}')

    # returns (uint256 lastFulfillmentRequestId, uint256 lastRequestId, uint256 lastCheckpointIndex);
    withdraw_queue_stat = withdraw.functions.getWithdrawQueueStat().call()
    logging.info(f'Dawn withdraw_queue_stat : {withdraw_queue_stat[0]},{withdraw_queue_stat[1]},{withdraw_queue_stat[2]}')

    latest_index = withdraw_queue_stat[0] + target_index

    full_metrics.lastRequestIdToBeFulfilled = latest_index
    full_metrics.ethAmountToLock = target_value
    logging.info(f'Dawn latest_index : {latest_index},target_value: {target_value}, ')

    logging.info(f'DawnPool validators visible on Beacon: {full_metrics.beaconValidators}')
    return full_metrics


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
    logging.info(f'Time delta: {datetime.timedelta(seconds=delta_seconds)} or {delta_seconds} s')
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
        f'transientValidators before:{previous.getTransientValidators()} after:{current.getTransientValidators()} change:{current.getTransientValidators() - previous.getTransientValidators()}'
    )
    logging.info(
        f'beaconBalance before:{previous.beaconBalance} after:{current.beaconBalance} change:{current.beaconBalance - previous.beaconBalance}'
    )
    logging.info(
        f'bufferedBalance before:{previous.bufferedBalance} after:{current.bufferedBalance} change:{current.bufferedBalance - previous.bufferedBalance}'
    )
    logging.info(
        f'transientBalance before:{previous.getTransientBalance()} after:{current.getTransientBalance()} change:{current.getTransientBalance() - previous.getTransientBalance()}'
    )
    logging.info(f'totalPooledEther before:{previous.getTotalPooledEther()} after:{current.getTotalPooledEther()} ')
    logging.info(f'activeValidatorBalance now:{current.activeValidatorBalance} ')

    # 验证者的币余额期望值(reward_base)  根据当前 epoch 中出现的有效验证者数量和每个验证者的抵押金额(DEPOSIT_SIZE) 计算出当前 epoch 的奖励基数 + 上一个epoch该验证者的余额
    reward_base = appeared_validators * DEPOSIT_SIZE + previous.beaconBalance
    # 验证者当前epoch应该获得的奖励 = 当前epoch结束时验证者余额 - 验证者的币余额期望值
    reward = current.beaconBalance - reward_base
    if not previous.getTotalPooledEther():
        logging.info(
            'The DawnPool has no funds under its control. Probably the system has been just deployed and has never been deposited'
        )
        return

    if not delta_seconds:
        logging.info('No time delta between current and previous epochs. Skip APR calculations.')
        assert reward == 0
        assert current.beaconValidators == previous.beaconValidators
        assert current.beaconBalance == current.beaconBalance
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
        logging.info(
            f'Rewards will increase Total pooled ethers by: {reward / previous.getTotalPooledEther() * 100:.4f} %'
        )
        logging.info(f'Daily staking reward rate for active validators: {daily_reward_rate * 100:.8f} %')
        logging.info(f'Staking APR for active validators: {apr * 100:.4f} %')
        if apr > current.MAX_APR:
            # warnings = True
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logging.warning('Staking APR too high! Talk to your fellow oracles before submitting!')
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if apr < current.MIN_APR:
            # warnings = True
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logging.warning('Staking APR too low! Talk to your fellow oracles before submitting!')
            logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    else:
        # warnings = True
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


def get_timestamp_by_epoch(beacon_spec, epoch_id):
    """Required to calculate time-bound values such as APR"""
    slots_per_epoch = beacon_spec[1]
    seconds_per_slot = beacon_spec[2]
    genesis_time = beacon_spec[3]
    return genesis_time + slots_per_epoch * seconds_per_slot * epoch_id
