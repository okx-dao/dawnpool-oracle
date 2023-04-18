import sys
from typing import cast

from prometheus_client import start_http_server
from web3.middleware import simple_cache_middleware

from src import variables
from src.metrics.healthcheck_server import start_pulse_server
from src.metrics.logging import logging
from src.metrics.prometheus.basic import ENV_VARIABLES_INFO, BUILD_INFO
from src.modules.accounting.accounting import Accounting
from src.modules.ejector.ejector import Ejector
from src.modules.checks.checks_module import ChecksModule
from src.typings import OracleModule
from src.utils.build import get_build_info
from src.web3py.extensions import (
    LidoContracts,
    TransactionUtils,
    ConsensusClientModule,
    KeysAPIClientModule,
    LidoValidatorsProvider,
    FallbackProviderModule
)
from src.web3py.middleware import metrics_collector
from src.web3py.typings import Web3

from src.web3py.contract_tweak import tweak_w3_contracts


logger = logging.getLogger()


def main(module: OracleModule):
    build_info = get_build_info()
    logger.info({
        'msg': 'Oracle startup.',
        'variables': {
            **build_info,
            'module': module,
            'ACCOUNT': variables.ACCOUNT.address if variables.ACCOUNT else 'Dry',
            'LIDO_LOCATOR_ADDRESS': variables.LIDO_LOCATOR_ADDRESS,
            'MAX_CYCLE_LIFETIME_IN_SECONDS': variables.MAX_CYCLE_LIFETIME_IN_SECONDS,
        },
    })
    ENV_VARIABLES_INFO.info({
        "ACCOUNT": str(variables.ACCOUNT.address) if variables.ACCOUNT else 'Dry',
        "LIDO_LOCATOR_ADDRESS": str(variables.LIDO_LOCATOR_ADDRESS),
        "FINALIZATION_BATCH_MAX_REQUEST_COUNT": str(variables.FINALIZATION_BATCH_MAX_REQUEST_COUNT),
        "MAX_CYCLE_LIFETIME_IN_SECONDS": str(variables.MAX_CYCLE_LIFETIME_IN_SECONDS),
    })
    BUILD_INFO.info(build_info)

    logger.info({'msg': f'Start healthcheck server for Docker container on port {variables.HEALTHCHECK_SERVER_PORT}'})
    start_pulse_server()

    logger.info({'msg': f'Start http server with prometheus metrics on port {variables.PROMETHEUS_PORT}'})
    start_http_server(variables.PROMETHEUS_PORT)

    logger.info({'msg': 'Initialize multi web3 provider.'})
    web3 = Web3(FallbackProviderModule(variables.EXECUTION_CLIENT_URI))

    logger.info({'msg': 'Modify web3 with custom contract function call.'})
    tweak_w3_contracts(web3)

    logger.info({'msg': 'Initialize consensus client.'})
    cc = ConsensusClientModule(variables.CONSENSUS_CLIENT_URI, web3)

    logger.info({'msg': 'Initialize keys api client.'})
    kac = KeysAPIClientModule(variables.KEYS_API_URI, web3)

    check_providers_chain_ids(web3, cc, kac)

    web3.attach_modules({
        'lido_contracts': LidoContracts,
        'lido_validators': LidoValidatorsProvider,
        'transaction': TransactionUtils,
        'cc': lambda: cc,  # type: ignore[dict-item]
        'kac': lambda: kac,  # type: ignore[dict-item]
    })

    logger.info({'msg': 'Add metrics middleware for ETH1 requests.'})
    web3.middleware_onion.add(metrics_collector)
    web3.middleware_onion.add(simple_cache_middleware)

    logger.info({'msg': 'Sanity checks.'})

    if module == OracleModule.ACCOUNTING:
        logger.info({'msg': 'Initialize Accounting module.'})
        accounting = Accounting(web3)
        accounting.check_contract_configs()
        accounting.run_as_daemon()
    elif module == OracleModule.EJECTOR:
        logger.info({'msg': 'Initialize Ejector module.'})
        ejector = Ejector(web3)
        ejector.check_contract_configs()
        ejector.run_as_daemon()


def check():
    logger.info({'msg': 'Check oracle is ready to work in the current environment.'})

    return ChecksModule().execute_module()


def check_providers_chain_ids(web3: Web3, cc: ConsensusClientModule, kac: KeysAPIClientModule):
    keys_api_chain_id = kac.check_providers_consistency()
    consensus_chain_id = cc.check_providers_consistency()
    execution_chain_id = cast(FallbackProviderModule, web3.provider).check_providers_consistency()

    if execution_chain_id == consensus_chain_id == keys_api_chain_id:
        return

    raise ValueError('Different chain ids detected:\n'
                     f'Execution chain id: {execution_chain_id}\n'
                     f'Consensus chain id: {consensus_chain_id}\n'
                     f'Keys API chain id: {keys_api_chain_id}\n')


if __name__ == '__main__':
    module_name = sys.argv[-1]
    if module_name not in iter(OracleModule):
        msg = f'Last arg should be one of {[str(item) for item in OracleModule]}, received {module_name}.'
        logger.error({'msg': msg})
        raise ValueError(msg)

    module = OracleModule(module_name)
    if module == OracleModule.CHECK:
        errors = variables.check_uri_required_variables()
        variables.raise_from_errors(errors)

        sys.exit(check())


    errors = variables.check_all_required_variables()
    variables.raise_from_errors(errors)
    main(module)
