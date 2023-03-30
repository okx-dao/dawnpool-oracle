import logging
from abc import ABC
from http import HTTPStatus
from typing import Optional, Tuple, Sequence, Callable
from urllib.parse import urljoin, urlparse

from prometheus_client import Histogram
from requests import Session, JSONDecodeError, Timeout
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from src.variables import HTTP_REQUEST_RETRY_COUNT, HTTP_REQUEST_SLEEP_BEFORE_RETRY_IN_SECONDS, HTTP_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class NoHostsProvided(Exception):
    pass


class NotOkResponse(Exception):
    status: int
    text: str

    def __init__(self, *args, status: int, text: str):
        self.status = status
        self.text = text
        super().__init__(*args)


class HTTPProvider(ABC):
    PROMETHEUS_HISTOGRAM: Histogram

    def __init__(self, hosts: list[str]):
        if not hosts:
            raise NoHostsProvided(f"No hosts provided for {self.__class__.__name__}")

        self.hosts = hosts

        retry_strategy = Retry(
            total=HTTP_REQUEST_RETRY_COUNT,
            status_forcelist=[418, 429, 500, 502, 503, 504],
            backoff_factor=HTTP_REQUEST_SLEEP_BEFORE_RETRY_IN_SECONDS,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @staticmethod
    def _urljoin(host, url):
        if not host.endswith('/'):
            host += '/'
        return urljoin(host, url)

    def _get(
        self,
        endpoint: str,
        path_params: Optional[Sequence[str | int]] = None,
        query_params: Optional[dict] = None,
        force_raise: Callable[..., Exception | None] = lambda _: None,
    ) -> Tuple[dict | list, dict]:
        """
        Get request with fallbacks
        Returns (data, meta) or raises exception

        force_raise - function that returns an Exception if it should be thrown immediately.
        Sometimes NotOk response from first provider is the response that we are expecting.
        """
        errors: list[Exception] = []

        for host in self.hosts:
            try:
                return self._get_without_fallbacks(host, endpoint, path_params, query_params)
            except Exception as e:  # pylint: disable=W0703
                errors.append(e)

                # Check if exception should be raised immediately
                if to_force_raise := force_raise(errors):
                    raise to_force_raise from e

                logger.warning(
                    {
                        'msg': f'[{self.__class__.__name__}] Host [{urlparse(host).netloc}] responded with error',
                        'error': str(e),
                        'provider': urlparse(host).netloc,
                    }
                )

        # Raise error from last provider.
        raise errors[-1]

    def _get_without_fallbacks(
        self,
        host: str,
        endpoint: str,
        path_params: Optional[Sequence[str | int]] = None,
        query_params: Optional[dict] = None
    ) -> Tuple[dict | list, dict]:
        """
        Simple get request without fallbacks
        Returns (data, meta) or raises exception
        """
        complete_endpoint = endpoint.format(*path_params) if path_params else endpoint

        with self.PROMETHEUS_HISTOGRAM.time() as t:
            try:
                response = self.session.get(
                    self._urljoin(host, complete_endpoint if path_params else endpoint),
                    params=query_params,
                    timeout=HTTP_REQUEST_TIMEOUT,
                )
            except Timeout as error:
                msg = f'Timeout error from {complete_endpoint}.'
                logger.debug({'msg': msg})
                t.labels(
                    endpoint=endpoint,
                    code=0,
                    domain=urlparse(host).netloc,
                )
                raise TimeoutError(msg) from error

            response_fail_msg = f'Response from {complete_endpoint} [{response.status_code}] with text: "{str(response.text)}" returned.'

            if response.status_code != HTTPStatus.OK:
                logger.debug({'msg': response_fail_msg})
                raise NotOkResponse(response_fail_msg, status=response.status_code, text=response.text)

            try:
                json_response = response.json()
            except JSONDecodeError as error:
                logger.debug({'msg': response_fail_msg})
                raise error
            finally:
                t.labels(
                    endpoint=endpoint,
                    code=response.status_code,
                    domain=urlparse(host).netloc,
                )

        if 'data' in json_response:
            data = json_response['data']
            del json_response['data']
            meta = json_response
        else:
            data = json_response
            meta = {}

        return data, meta
