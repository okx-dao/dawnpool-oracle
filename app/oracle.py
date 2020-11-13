import json
import logging
import os
import time

from web3 import Web3, WebsocketProvider, HTTPProvider

from beacon import get_beacon
from contracts import get_validators_keys, get_report_interval

logging.basicConfig(
    level=logging.INFO, format='%(levelname)8s %(asctime)s <daemon> %(message)s', datefmt='%m-%d %H:%M:%S'
)

envs = [
    'ETH1_NODE',
    'ETH2_NODE',
    'ORACLE_CONTRACT',
    'ORACLE_ABI_FILE',
    'POOL_ABI_FILE',
    'REGISTRY_ABI_FILE',
    'MANAGER_PRIV_KEY',
]
missing = []
for env in envs:
    if env not in os.environ:
        missing.append(env)
        logging.error('Variable %s is missing', env)

if missing:
    exit(1)

eth1_provider = os.environ['ETH1_NODE']
eth2_provider = os.environ['ETH2_NODE']
oracle_address = os.environ['ORACLE_CONTRACT']
if not Web3.isChecksumAddress(oracle_address):
    oracle_address = Web3.toChecksumAddress(oracle_address)
oracle_abi_path = os.environ['ORACLE_ABI_FILE']
pool_abi_path = os.environ['POOL_ABI_FILE']
registry_abi_path = os.environ['REGISTRY_ABI_FILE']
manager_privkey = os.environ['MANAGER_PRIV_KEY']

GAS_LIMIT = int(os.getenv('GAS_LIMIT', 1000000))

if eth1_provider.startswith('http'):
    provider = HTTPProvider(eth1_provider)
elif eth1_provider.startswith('ws'):
    provider = WebsocketProvider(eth1_provider)
else:
    logging.error('Unsupported ETH provider!')
    exit(1)

w3 = Web3(provider)

if not w3.isConnected():
    logging.error('ETH node connection error!')
    exit(1)

w3.eth.defaultAccount = w3.eth.account.privateKeyToAccount(manager_privkey)

with open(oracle_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
oracle = w3.eth.contract(abi=abi['abi'], address=oracle_address)

pool_address = oracle.functions.pool().call({'from': w3.eth.defaultAccount.address})

with open(pool_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
pool = w3.eth.contract(abi=abi['abi'], address=pool_address)

registry_address = pool.functions.getOperators().call({'from': w3.eth.defaultAccount.address})

with open(registry_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)
registry = w3.eth.contract(abi=abi['abi'], address=registry_address)

# temporary solution
oracle_abi_path = os.environ['NEW_ORACLE_ABI_FILE']
oracle_address = '0xa2BE6439d8def6dD6523AeFd02a1356772d15569'
with open(oracle_abi_path, 'r') as file:
    a = file.read()
abi = json.loads(a)

oracle = w3.eth.contract(abi=abi['abi'], address=oracle_address)
# end here

beacon_spec = oracle.functions.beaconSpec().call({'from': w3.eth.defaultAccount.address})

slots_per_epoch = beacon_spec[0]
seconds_per_slot = beacon_spec[1]

# reportable_epoch = oracle.functions.getCurrentReportableEpoch.call({'from': w3.eth.defaultAccount.address})
reportable_epoch = 1

beacon = get_beacon(eth2_provider, slots_per_epoch)

logging.info('=====The oracle daemon is started!=====')
logging.info('============ CONFIGURATION ============')
logging.info(f'ETH1 Node: {eth1_provider}')
logging.info(f'ETH2 Node: {eth2_provider}')
logging.info('Connecting to %s', beacon.__class__.__name__)
logging.info(f'Oracle contract address: {oracle_address}')
logging.info(f'Registry contract address: {registry_address}')
logging.info(f'Manager account: {w3.eth.defaultAccount.address}')
logging.info(f'Seconds per slot: {seconds_per_slot}')
logging.info(f'Slots per epoch: {slots_per_epoch}')
logging.info('=======================================')


await_time_const = 5
while True:
    # Get actual slot and last finalized slot from beacon head data
    last_slots = beacon.get_actual_slot()
    current_epoch = int(last_slots['finalized_slot'] / slots_per_epoch)
    if reportable_epoch <= current_epoch:
        validators_keys = get_validators_keys(registry, w3)
        if len(validators_keys) == 0:
            logging.warning('No keys on Staking Providers Registry contract')
        # Get sum of balances
        sum_balance = beacon.get_balances(reportable_epoch, validators_keys)
        tx_hash = oracle.functions.reportBeacon(reportable_epoch, sum_balance).buildTransaction(
            {'from': w3.eth.defaultAccount.address, 'gas': GAS_LIMIT}
        )
        tx_hash['nonce'] = w3.eth.getTransactionCount(
            w3.eth.defaultAccount.address
        )  # Get correct transaction nonce for sender from the node
        signed = w3.eth.account.signTransaction(tx_hash, w3.eth.defaultAccount.privateKey)
        tx_hash = w3.eth.sendRawTransaction(signed.rawTransaction)
        logging.info('Transaction in progress...')
        tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
        if tx_receipt.status == 1:
            logging.info('Transaction successful')
            logging.info('Balances pushed!')
        else:
            logging.warning('Transaction reverted')
            logging.warning(tx_receipt)
            # TODO logic when transaction reverted
    time.sleep(await_time_const)
