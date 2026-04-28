from azure.core.exceptions import HttpResponseError, ServiceResponseError, ServiceRequestError
from httpcore import RemoteProtocolError as HTTPCoreError
import httpx
import requests
import time

from src.regrets.sum_call_seq import get_summary as _get_summary
from src.regrets.optimal_rand_seq_tele import opt_eval as _opt_eval

def azure_retry(func):
    def wrapper(*args, **kwargs):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                # throttle to avoid bursting the API
                time.sleep(0.2)
                return result

            except HttpResponseError as e:
                # 429 Too Many Requests
                if e.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    print(f"[Azure 429] retry #{attempt}/{max_retries} after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                raise

            except (ServiceResponseError, ServiceRequestError) as e:
                msg = str(e).lower()
                # catch read‐timeout errors from the Azure SDK
                if "timed out" in msg or "timeout" in msg:
                    if attempt == max_retries:
                        raise
                    backoff = 2 ** (attempt - 1)
                    print(f"[Azure timeout] retry #{attempt}/{max_retries} after {backoff}s: {e}")
                    time.sleep(backoff)
                    continue
                raise

            except requests.exceptions.ReadTimeout as e:
                # lower‐level requests timeout
                if attempt == max_retries:
                    raise
                backoff = 2 ** (attempt - 1)
                print(f"[requests ReadTimeout] retry #{attempt}/{max_retries} after {backoff}s: {e}")
                time.sleep(backoff)
                continue

            except httpx.RemoteProtocolError as e:
                # streaming got cut off; exponential backoff
                if attempt == max_retries:
                    raise
                backoff = 2 ** (attempt - 1)
                print(f"[HTTPX stream error] retry #{attempt}/{max_retries} after {backoff}s: {e}")
                time.sleep(backoff)
                continue

        raise RuntimeError(f"Azure retries exhausted for {func.__name__}")
    return wrapper

def get_summary():
    return azure_retry(_get_summary)

def opt_eval():
    return azure_retry(_opt_eval)