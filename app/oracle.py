# SPDX-FileCopyrightText: 2020 Lido <info@lido.fi>

# SPDX-License-Identifier: GPL-3.0

import json
import logging
import os
import datetime
import time
from typing import Tuple

from urllib3.exceptions import MaxRetryError
from web3_multi_provider import MultiProvider

from exceptions import BeaconConnectionTimeoutException

from prometheus_client import start_http_server
from web3 import Web3
from web3.exceptions import SolidityError, CannotHandleRequest, TimeExhausted

from beacon import BeaconChainClient
from log import init_log
from metrics import compare_pool_metrics, get_previous_metrics, get_light_current_metrics, get_full_current_metrics
from prometheus_metrics import metrics_exporter_state
from state_proof import encode_proof_data

init_log(stdout_level=os.environ.get('LOG_LEVEL_STDOUT', 'INFO'))
logger = logging.getLogger()

meta_envs = ['VERSION', 'COMMIT_MESSAGE', 'COMMIT_HASH', 'COMMIT_DATETIME', 'BUILD_DATETIME', 'TAGS', 'BRANCH']

envs = [
    'WEB3_PROVIDER_URI',
    'BEACON_NODE',
    'POOL_CONTRACT',
    'REWARDS_VAULT_ADDRESS',
]
if os.getenv('FORCE'):
    logging.error(
        'The flag "FORCE" is obsolete in favour of '
        '"FORCE_DO_NOT_USE_IN_PRODUCTION", '
        'please NEVER use it in production'
    )
    exit(1)

missing = []
for env in envs:
    if env not in os.environ or os.environ[env] == '':
        missing.append(env)
        logging.error('Mandatory variable %s is missing', env)

if missing:
    exit(1)

ARTIFACTS_DIR = './assets'
ORACLE_ARTIFACT_FILE = 'DawnPoolOracle.json'
POOL_ARTIFACT_FILE = 'DawnDeposit.json'
REGISTRY_ARTIFACT_FILE = 'DepositNodeManager.json'
BURNER_ARTIFACT_FILE = 'Burner.json'
WITHDRAW_ARTIFACT_FILE = 'DawnWithdraw.json'

DEFAULT_SLEEP = 60
DEFAULT_COUNTDOWN_SLEEP = 10
DEFAULT_GAS_LIMIT = 1_500_000

prometheus_metrics_port = int(os.getenv('PROMETHEUS_METRICS_PORT', 8000))

# 奖励库的地址
rewards_vault_address = os.environ['REWARDS_VAULT_ADDRESS']
if not Web3.isChecksumAddress(rewards_vault_address):
    rewards_vault_address = Web3.toChecksumAddress(rewards_vault_address)

eth1_provider = os.environ['WEB3_PROVIDER_URI']
beacon_provider = os.environ['BEACON_NODE']

pool_address = os.environ['POOL_CONTRACT']
if not Web3.isChecksumAddress(pool_address):
    pool_address = Web3.toChecksumAddress(pool_address)

oracle_address = os.environ['ORACLE_CONTRACT']
if not Web3.isChecksumAddress(oracle_address):
    oracle_address = Web3.toChecksumAddress(oracle_address)

node_manager_address = os.environ['NODE_MANAGER_CONTRACT']
if not Web3.isChecksumAddress(node_manager_address):
    node_manager_address = Web3.toChecksumAddress(node_manager_address)

burner_address = os.environ['BURNER_CONTRACT']
if not Web3.isChecksumAddress(burner_address):
    burner_address = Web3.toChecksumAddress(burner_address)

withdraw_address = os.environ['WITHDRAW_CONTRACT']
if not Web3.isChecksumAddress(withdraw_address):
    withdraw_address = Web3.toChecksumAddress(withdraw_address)

# 获取合约abi路径
oracle_abi_path = os.path.join(ARTIFACTS_DIR, ORACLE_ARTIFACT_FILE)
pool_abi_path = os.path.join(ARTIFACTS_DIR, POOL_ARTIFACT_FILE)
registry_abi_path = os.path.join(ARTIFACTS_DIR, REGISTRY_ARTIFACT_FILE)
burner_abi_path = os.path.join(ARTIFACTS_DIR, BURNER_ARTIFACT_FILE)
withdraw_abi_path = os.path.join(ARTIFACTS_DIR, WITHDRAW_ARTIFACT_FILE)

member_privkey = os.getenv('MEMBER_PRIV_KEY')
SLEEP = int(os.getenv('SLEEP', DEFAULT_SLEEP))
COUNTDOWN_SLEEP = int(os.getenv('COUNTDOWN_SLEEP', DEFAULT_COUNTDOWN_SLEEP))

run_as_daemon = int(os.getenv('DAEMON', 0))
force = int(os.getenv('FORCE_DO_NOT_USE_IN_PRODUCTION', 0))

# 用户私钥
dry_run = member_privkey is None

GAS_LIMIT = int(os.getenv('GAS_LIMIT', DEFAULT_GAS_LIMIT))

# 数据块的块号
ORACLE_FROM_BLOCK = int(os.getenv('ORACLE_FROM_BLOCK', 0))
#  指定一个epoch作为提款请求的启示点 在这个epoch之前，如果还有其他的验证节点需要提款处理，则这些提款请求也将被考虑在内。如果没有指定该参数，则默认从当前epoch开始处理验证节点的提款请求。
consider_withdrawals_from_epoch = os.environ.get('CONSIDER_WITHDRAWALS_FROM_EPOCH')

provider = MultiProvider(eth1_provider.split(','))
w3 = Web3(provider)

if not w3.isConnected():
    logging.error('ETH node connection error!')
    exit(1)

# See EIP-155 for the list of other well-known Net IDs
networks = {
    1: {'name': 'Mainnet', 'engine': 'PoW'},
    5: {'name': 'Goerli', 'engine': 'PoA'},
    # 1337: {'name': 'E2E', 'engine': 'PoA'},
}

network_id = w3.eth.chain_id
if network_id in networks.keys():
    logging.info(f"Connected to {networks[network_id]['name']} network ({networks[network_id]['engine']} engine)")
    if networks[network_id]['engine'] == 'PoA':
        logging.info("Injecting PoA compatibility middleware")
        # MultiProvider already supports PoA, so no need to inject manually
        # from web3.middleware import geth_poa_middleware
        # w3.middleware_onion.inject(geth_poa_middleware, layer=0)

if dry_run:
    logging.info('MEMBER_PRIV_KEY not provided, running in read-only (DRY RUN) mode')
else:
    logging.info('MEMBER_PRIV_KEY provided, running in transactable (PRODUCTION) mode')
    account = w3.eth.account.from_key(member_privkey)
    logging.info(f'Member account: {account.address}')

# Get Pool contract
with open(pool_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
pool = w3.eth.contract(abi=abi['abi'], address=pool_address)  # contract object
# Get Oracle contract
# oracle_address = pool.functions.getOracle().call()  # oracle contract
# logger.info(f'{oracle_address=}')

with open(oracle_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
oracle = w3.eth.contract(abi=abi['abi'], address=oracle_address)

with open(registry_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
registry = w3.eth.contract(abi=abi['abi'], address=node_manager_address)

with open(burner_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
burner = w3.eth.contract(abi=abi['abi'], address=burner_address)

with open(withdraw_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
withdraw = w3.eth.contract(abi=abi['abi'], address=withdraw_address)

# Get Registry contract
# registry_address = pool.functions.getOperators().call()
# logger.info(f'{registry_address=}')
#
# with open(registry_abi_path, 'r') as file:
#     a = file.read()
# abi = json.loads(a)
# registry = w3.eth.contract(abi=abi['abi'], address=registry_address)


# Get Beacon specs from contract 查询了当前区块链上的信标链规范数据
beacon_spec = oracle.functions.getBeaconSpec().call()
# 每个信标链帧（Beacon Chain Frame）中包含的epochs: 每一帧包含的epochs在oracle中定义 225个
epochs_per_frame = beacon_spec[0]
# 每个epoch的插槽数: 32个slot
slots_per_epoch = beacon_spec[1]
# 每个槽位的时间长度：12秒（Seconds Per Slot）
seconds_per_slot = beacon_spec[2]
# 以太坊2的创世纪时间戳 主网: 1606824023  Goerli测试网的值为：1616508000
genesis_time = beacon_spec[3]

beacon = BeaconChainClient(beacon_provider, slots_per_epoch)

if run_as_daemon:
    # 以守护进程模式运行（无限循环）
    logging.info('DAEMON=1 Running in daemon mode (in endless loop).')
else:
    # 单次迭代模式运行（报完会退出）
    logging.info('DAEMON=0 Running in single iteration mode (will exit after reporting).')

if force:
    logging.info('FORCE_DO_NOT_USE_IN_PRODUCTION=1 Running in enforced mode.')
    # 在强制模式下，TX 总是被发送，即使它看起来可疑。 切勿在生产中使用它！
    logging.warning("In enforced mode TX gets always sent even if it looks suspicious. NEVER use it in production!")
else:
    logging.info('FORCE_DO_NOT_USE_IN_PRODUCTION=0')

logging.info(f'SLEEP={SLEEP} s (pause between iterations in DAEMON mode)')
logging.info(f'GAS_LIMIT={GAS_LIMIT} gas units')
logging.info(f'POOL_CONTRACT={pool_address}')

logging.info(f'Oracle contract address: {oracle_address} (auto-discovered)')
logging.info(f'Registry contract address: {node_manager_address} (auto-discovered)')
logging.info(f'Seconds per slot: {seconds_per_slot} (auto-discovered)')
logging.info(f'Slots per epoch: {slots_per_epoch} (auto-discovered)')
logging.info(f'Epochs per frame: {epochs_per_frame} (auto-discovered)')
logging.info(f'Genesis time: {genesis_time} (auto-discovered)')


def build_report_beacon_tx(epoch, balance, validators, rewardsBalance, exitedValidatorsCount, burnedPethAmount,
                           lastRequestIdToBeFulfilled, ethAmountToLock):  # hash tx
    max_fee_per_gas, max_priority_fee_per_gas = _get_tx_gas_params()

    data = {
        'epochId': epoch,
        'beaconBalance': balance,
        'beaconValidators': validators,
        'rewardsVaultBalance': rewardsBalance,
        'exitedValidators': exitedValidatorsCount,
        'burnedPEthAmount': burnedPethAmount,
        'lastRequestIdToBeFulfilled': lastRequestIdToBeFulfilled,
        'ethAmountToLock': ethAmountToLock
    }

    # todo 金额位数修改
    return oracle.functions.reportBeacon(data).buildTransaction(
        {
            'from': account.address,
            'gas': GAS_LIMIT,
            'maxFeePerGas': max_fee_per_gas,
            'maxPriorityFeePerGas': max_priority_fee_per_gas,
        }
    )


def sign_and_send_tx(tx):
    logging.info('Preparing TX... CTRL-C to abort')
    time.sleep(3)  # To be able to Ctrl + C
    tx['nonce'] = w3.eth.getTransactionCount(account.address)  # Get correct transaction nonce for sender from the node
    signed = w3.eth.account.sign_transaction(tx, account.key)
    logging.info(f'TX hash: {signed.hash.hex()} ... CTRL-C to abort')
    time.sleep(3)
    logging.info('Sending TX... CTRL-C to abort')
    time.sleep(3)
    tx_hash = w3.eth.sendRawTransaction(signed.rawTransaction)
    logging.info('TX has been sent. Waiting for receipt...')
    tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
    if tx_receipt.status == 1:
        logging.info('TX successful')
        metrics_exporter_state.txSuccess.observe(1)
    else:
        logging.warning('TX reverted')
        logging.warning(tx_receipt)
        metrics_exporter_state.txRevert.observe(1)


def prompt(prompt_message, prompt_end):
    print(prompt_message, end='')
    while True:
        choice = input().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print('Please respond with [y or n]: ', end=prompt_end)
            continue


def main():
    if not consider_withdrawals_from_epoch:
        raise ValueError('CONSIDER_WITHDRAWALS_FROM_EPOCH is not set')

    logger.info(f'start prometheus metrics server on the port: {prometheus_metrics_port}')
    start_http_server(prometheus_metrics_port)

    logging.info('Starting the main loop')
    while True:
        try:
            run_once()
            sleep()
        except StopIteration:
            break
        except CannotHandleRequest as exc:
            if 'Could not discover provider while making request: method:eth_chainId' in str(exc):
                logger.exception("handle 'Could not discover provider' problem")
                time.sleep(1)
                continue
            else:
                raise
        except (BeaconConnectionTimeoutException, MaxRetryError) as exc:
            if run_as_daemon:
                logging.exception(exc)
                metrics_exporter_state.beaconNodeTimeoutCount.inc()
                continue
            else:
                raise
        except ValueError as exc:
            (args,) = exc.args
            if run_as_daemon and args["code"] == -32000:
                logging.exception(exc)
                metrics_exporter_state.underpricedExceptionsCount.inc()
                continue
            else:
                raise
        except TimeExhausted as exc:
            if run_as_daemon:
                logging.exception(exc)
                metrics_exporter_state.timeExhaustedExceptionsCount.inc()
                continue
            else:
                raise
        except Exception as exc:
            if run_as_daemon:
                logging.exception(exc)
                metrics_exporter_state.exceptionsCount.inc()
            else:
                raise


def run_once():
    update_beacon_data()

    if not run_as_daemon:
        logging.info('We are in single-iteration mode, so exiting. Set DAEMON=1 env to run in the loop.')
        raise StopIteration()

    logging.info(f'We are in DAEMON mode. Sleep {SLEEP} s and continue')


def update_beacon_data():
    # Get previously reported data 获取之前上报的数据
    prev_metrics = get_previous_metrics(w3, pool, oracle, beacon_spec, rewards_vault_address, ORACLE_FROM_BLOCK)
    metrics_exporter_state.set_prev_pool_metrics(prev_metrics)
    if prev_metrics:
        logging.info(f'Previously reported epoch: {prev_metrics.epoch}')
        logging.info(
            f'Previously reported beaconBalance: {prev_metrics.beaconBalance} wei or {prev_metrics.beaconBalance / 1e18} ETH'
        )
        logging.info(
            f'Previously reported bufferedBalance: {prev_metrics.bufferedBalance} wei or {prev_metrics.bufferedBalance / 1e18} ETH'
        )
        logging.info(f'Previous validator metrics: depositedValidators:{prev_metrics.depositedValidators}')
        logging.info(f'Previous validator metrics: transientValidators:{prev_metrics.getTransientValidators()}')
        logging.info(f'Previous validator metrics: beaconValidators:{prev_metrics.beaconValidators}')
        logging.info(f'Previous validator metrics: rewardsVaultBalance:{prev_metrics.rewardsVaultBalance}')
        logging.info(
            f'Timestamp of previous report: {datetime.datetime.fromtimestamp(prev_metrics.timestamp)} or {prev_metrics.timestamp}'
        )

    # Get minimal metrics that are available without polling 获取当前信标链的性能度量（Metrics）
    current_metrics = get_light_current_metrics(w3, beacon, pool, oracle, beacon_spec)
    metrics_exporter_state.set_current_pool_metrics(current_metrics)

    logging.info(
        f'Currently Metrics epoch: {current_metrics.epoch} Prev Metrics epoch {prev_metrics.epoch} '
    )

    # 一天225个epoch 如果当前epoch <= 上次提交的epoch加一天 说明一天内已经提交过 不提交
    if current_metrics.epoch <= prev_metrics.epoch:  # commit happens once per day
        logging.info(f'Currently reportable epoch {current_metrics.epoch} has already been reported. Skipping it.')
        return

    # Get full metrics using polling (get keys from registry, get balances from beacon)
    current_metrics = get_full_current_metrics(
        w3, pool, registry, burner, withdraw, beacon, beacon_spec, current_metrics, rewards_vault_address
    )
    metrics_exporter_state.set_current_pool_metrics(current_metrics)
    # 对比
    warnings = compare_pool_metrics(prev_metrics, current_metrics)

    logging.info(
        f'Tx call data: oracle.reportBeacon({current_metrics.epoch}, {current_metrics.beaconBalance}, {current_metrics.beaconValidators}, {current_metrics.rewardsVaultBalance}'
        f', {current_metrics.exitedValidatorsCount} , {current_metrics.burnedPethAmount}, {current_metrics.lastRequestIdToBeFulfilled}, {current_metrics.ethAmountToLock})'
    )
    # 上报数据
    if not dry_run:

        try:
            metrics_exporter_state.reportableFrame.set(True)
            tx = build_report_beacon_tx(
                current_metrics.epoch, current_metrics.beaconBalance, current_metrics.beaconValidators,
                current_metrics.rewardsVaultBalance,
                current_metrics.exitedValidatorsCount, current_metrics.burnedPethAmount,
                current_metrics.lastRequestIdToBeFulfilled, current_metrics.ethAmountToLock)
            # Create the tx and execute it locally to check validity
            # logging.info(f'Calling tx: ', {tx})
            w3.eth.call(tx)

            logging.info('Calling tx locally succeeded.')
            if run_as_daemon:
                if warnings:
                    if force:
                        sign_and_send_tx(tx)
                    else:
                        logging.warning('Cannot report suspicious data in DAEMON mode for safety reasons.')
                        logging.warning('You can submit it interactively (with DAEMON=0) and interactive [y/n] prompt.')
                        logging.warning(
                            "In DAEMON mode it's possible with enforcement flag (FORCE_DO_NOT_USE_IN_PRODUCTION=1). Never use it in production."
                        )
                else:
                    sign_and_send_tx(tx)
            else:
                print(f'Tx data: {tx.__repr__()}')
                if prompt('Should we send this TX? [y/n]: ', ''):
                    sign_and_send_tx(tx)

        except SolidityError as sl:
            str_sl = str(sl)

            if "EPOCH_IS_TOO_OLD" in str_sl:
                logging.info('Calling tx locally reverted "EPOCH_IS_TOO_OLD"')
            elif "ALREADY_SUBMITTED" in str_sl:
                logging.info('Calling tx locally reverted "ALREADY_SUBMITTED"')
            elif "EPOCH_HAS_NOT_YET_BEGUN" in str_sl:
                logging.info('Calling tx locally reverted "EPOCH_HAS_NOT_YET_BEGUN"')
            elif "MEMBER_NOT_FOUND" in str_sl:
                logging.warning(
                    'Calling tx locally reverted "MEMBER_NOT_FOUND". Maybe you are using the address that is not in the members list?'
                )
            elif "REPORTED_MORE_DEPOSITED" in str_sl:
                logging.warning(
                    'Calling tx locally reverted "REPORTED_MORE_DEPOSITED". Something wrong with calculated balances on the beacon or the validators list'
                )
            elif "REPORTED_LESS_VALIDATORS" in str_sl:
                logging.warning(
                    'Calling tx locally reverted "REPORTED_LESS_VALIDATORS". Oracle can\'t report less validators than seen on the Beacon before.'
                )
            else:
                logging.error(f'Calling tx locally failed: {str_sl}')
        except ValueError as exc:
            (args,) = exc.args
            if args["code"] == -32000:
                raise
            else:
                metrics_exporter_state.exceptionsCount.inc()
                logging.exception(f'Unexpected exception. {type(exc)}')
        except TimeExhausted:
            raise
        except Exception as exc:
            metrics_exporter_state.exceptionsCount.inc()
            logging.exception(f'Unexpected exception. {type(exc)}')

    else:
        logging.info('The tx hasn\'t been actually sent to the oracle contract! We are in DRY RUN mode')
        logging.info('Provide MEMBER_PRIV_KEY to be able to transact')


def sleep():
    # sleep and countdown
    awake_at = time.time() + SLEEP
    while time.time() < awake_at:
        time.sleep(COUNTDOWN_SLEEP)
        countdown = awake_at - time.time()
        if countdown < 0:
            break
        metrics_exporter_state.reportableFrame.set(False)
        metrics_exporter_state.daemonCountDown.set(countdown)
        blocknumber = w3.eth.getBlock('latest')['number']
        metrics_exporter_state.nowEthV1BlockNumber.set(blocknumber)
        finalized_epoch_beacon = beacon.get_finalized_epoch()
        metrics_exporter_state.finalizedEpoch.set(finalized_epoch_beacon)

        logger.info(f'{awake_at=} {countdown=} {blocknumber=} {finalized_epoch_beacon=}')


def _get_tx_gas_params() -> Tuple[int, int]:
    """Return tx gas fee and priority fee"""
    base_fee_per_gas = w3.eth.get_block('latest').baseFeePerGas
    max_priority_fee_per_gas = w3.eth.max_priority_fee * 2
    max_fee_per_gas = int(base_fee_per_gas * 2 + max_priority_fee_per_gas)
    return max_fee_per_gas, max_priority_fee_per_gas


if __name__ == '__main__':
    main()
