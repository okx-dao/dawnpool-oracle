from dataclasses import dataclass

from eth_typing import ChecksumAddress, HexStr

from src.utils.dataclass import FromResponse


@dataclass
class LidoKey(FromResponse):
    key: HexStr
    depositSignature: HexStr
    operatorIndex: int
    used: bool
    moduleAddress: ChecksumAddress


# event SigningKeyExiting(uint256 indexed validatorId, address indexed operator, bytes pubkey);
@dataclass
class DawnPoolKey(FromResponse):
    pubkey: HexStr
    validatorId: int
    operator: HexStr


@dataclass
class KeysApiStatus(FromResponse):
    appVersion: str
    chainId: int
